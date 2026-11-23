"""
Offline evaluation for comparing recommenders.

Each user's most recent ratings are held out as a test set. A recommender
only sees the training ratings, and we check how many of its top-k
recommendations are movies the user went on to rate positively.
"""

import numpy as np
import pandas as pd

from src.data import GENRE_COLS


def train_test_split_by_time(ratings, test_fraction=0.2):
    """
    Hold out each user's most recent ratings.

    Splitting by time (instead of randomly) mimics the real task:
    use what a user has rated so far to predict what they'll like next.

    Args:
        ratings: DataFrame containing user_id, movie_id, rating, timestamp.
        test_fraction: Share of each user's ratings to hold out. Every user
            with at least 2 ratings gets at least 1 test rating and keeps
            at least 1 training rating.

    Returns:
        (train, test) DataFrames.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    ordered = ratings.sort_values(
        ["user_id", "timestamp", "movie_id"], kind="stable"
    )

    group = ordered.groupby("user_id")
    position = group.cumcount()
    size = group["movie_id"].transform("size")

    n_test = np.maximum(1, np.round(size * test_fraction)).clip(upper=size - 1)
    is_test = position >= size - n_test

    train = ordered[~is_test].reset_index(drop=True)
    test = ordered[is_test].reset_index(drop=True)
    return train, test


def evaluate(model, train, test, k=10, threshold=4, users=None):
    """
    Score one recommender, user by user.

    Args:
        model: Any object with recommend(ratings, user_id, n, threshold)
            that returns a DataFrame with a movie_id column.
        train: Ratings the model is allowed to see.
        test: Held-out ratings.
        k: Number of recommendations to score.
        threshold: Ratings at or above this count as "liked".
        users: Optional list of user_ids to evaluate (default: every user
            who liked at least one held-out movie).

    Returns:
        DataFrame with one row per user: precision, recall, hit,
        ndcg, mrr, and the list of recommended movie_ids.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if not 1 <= threshold <= 5:
        raise ValueError("Threshold must be between 1 and 5.")

    liked = (
        test[test["rating"] >= threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
    )

    train_users = set(train["user_id"])
    if users is None:
        users = [u for u in liked.index if u in train_users]

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    rows = []
    for user_id in users:
        relevant = liked.get(user_id, set())
        if not relevant or user_id not in train_users:
            continue

        recs = model.recommend(
            train, user_id=user_id, n=k, threshold=threshold
        )["movie_id"].tolist()[:k]

        hits = len(relevant.intersection(recs))
        dcg = sum(
            discounts[idx]
            for idx, movie_id in enumerate(recs)
            if movie_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = float(discounts[:ideal_hits].sum())
        mrr = next(
            (1.0 / rank for rank, movie_id in enumerate(recs, start=1) if movie_id in relevant),
            0.0,
        )

        rows.append({
            "user_id": user_id,
            "precision": hits / k,
            "recall": hits / len(relevant),
            "hit": hits > 0,
            "ndcg": dcg / idcg if idcg > 0 else 0.0,
            "mrr": mrr,
            "recommended": recs,
        })

    return pd.DataFrame(
        rows,
        columns=["user_id", "precision", "recall", "hit", "ndcg", "mrr", "recommended"]
    )


def _compute_personalization(recommended_lists):
    """
    Measure average recommendation diversity across users via 1 - cosine similarity.
    """
    non_empty = [list(dict.fromkeys(recs)) for recs in recommended_lists if len(recs) > 0]
    u_count = len(non_empty)
    if u_count < 2:
        return 0.0

    item_to_col = {}
    for recs in non_empty:
        for movie_id in recs:
            if movie_id not in item_to_col:
                item_to_col[movie_id] = len(item_to_col)

    matrix = np.zeros((u_count, len(item_to_col)), dtype=float)
    for row_idx, recs in enumerate(non_empty):
        cols = [item_to_col[m] for m in recs]
        matrix[row_idx, cols] = 1.0

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / norms
    sim = normed @ normed.T
    mean_off_diag = (sim.sum() - u_count) / (u_count * (u_count - 1))
    return float(np.clip(1.0 - mean_off_diag, 0.0, 1.0))


def compare_models(models, train, test, k=10, threshold=4, users=None):
    """
    Evaluate several recommenders on the same split.

    Args:
        models: Dict mapping a display name to a recommender.

    Returns:
        DataFrame with one row per model: users_evaluated, precision@k,
        recall@k, hit_rate@k, ndcg@k, mrr@k, coverage, and personalization.
    """
    catalog_size = pd.concat([train["movie_id"], test["movie_id"]]).nunique()

    summary = []
    for name, model in models.items():
        results = evaluate(
            model, train, test, k=k, threshold=threshold, users=users
        )
        recommended = {m for recs in results["recommended"] for m in recs}
        summary.append({
            "model": name,
            "users_evaluated": len(results),
            f"precision@{k}": results["precision"].mean(),
            f"recall@{k}": results["recall"].mean(),
            f"hit_rate@{k}": results["hit"].mean(),
            f"ndcg@{k}": results["ndcg"].mean(),
            f"mrr@{k}": results["mrr"].mean(),
            "coverage": len(recommended) / catalog_size if catalog_size > 0 else 0.0,
            "personalization": _compute_personalization(results["recommended"]),
        })

    return pd.DataFrame(summary).set_index("model")


def evaluate_by_user_activity(models, train, test, k=10, threshold=4, quantiles=3):
    """
    Break down model evaluation metrics across user training-activity tiers.

    Args:
        models: Dict mapping a display name to a recommender.
        train: Training ratings DataFrame.
        test: Held-out test ratings DataFrame.
        k: Top-k cutoff.
        threshold: Inclusive positive rating threshold.
        quantiles: Number of user activity buckets (default 3: Light/Medium/Heavy).

    Returns:
        DataFrame summarizing metrics for each (segment, model) pair.
    """
    if quantiles < 1:
        raise ValueError("quantiles must be at least 1.")

    liked_users = set(test.loc[test["rating"] >= threshold, "user_id"])
    train_counts = train.groupby("user_id").size()
    eligible_users = sorted(u for u in liked_users if u in train_counts.index)

    columns = [
        "segment",
        "model",
        "users_evaluated",
        "min_train_ratings",
        "max_train_ratings",
        f"precision@{k}",
        f"recall@{k}",
        f"hit_rate@{k}",
        f"ndcg@{k}",
        f"mrr@{k}",
    ]
    if not eligible_users:
        return pd.DataFrame(columns=columns)

    counts = train_counts.loc[eligible_users]
    tier_ids = pd.qcut(counts, q=quantiles, labels=False, duplicates="drop")
    n_bins = int(tier_ids.nunique())

    if n_bins == 3:
        tier_names = ["Light", "Medium", "Heavy"]
    elif n_bins == 2:
        tier_names = ["Light", "Heavy"]
    elif n_bins == 1:
        tier_names = ["All"]
    else:
        tier_names = [f"Tier {i + 1}" for i in range(n_bins)]

    sorted_bins = sorted(tier_ids.unique())
    bin_to_label = {b: tier_names[idx] for idx, b in enumerate(sorted_bins)}
    user_segment = tier_ids.map(bin_to_label)

    rows = []
    for model_name, model in models.items():
        results = evaluate(
            model, train, test, k=k, threshold=threshold, users=eligible_users
        )
        results["segment"] = results["user_id"].map(user_segment)
        for seg_label in [bin_to_label[b] for b in sorted_bins]:
            seg_res = results[results["segment"] == seg_label]
            seg_counts = counts.loc[seg_res["user_id"]]
            rows.append({
                "segment": seg_label,
                "model": model_name,
                "users_evaluated": len(seg_res),
                "min_train_ratings": int(seg_counts.min()),
                "max_train_ratings": int(seg_counts.max()),
                f"precision@{k}": seg_res["precision"].mean(),
                f"recall@{k}": seg_res["recall"].mean(),
                f"hit_rate@{k}": seg_res["hit"].mean(),
                f"ndcg@{k}": seg_res["ndcg"].mean(),
                f"mrr@{k}": seg_res["mrr"].mean(),
            })

    return pd.DataFrame(rows, columns=columns)


def user_genre_profile(ratings, movies, user_id, threshold=4):
    """
    Summarize a user's rating counts and positively rated genre frequencies.
    """
    if not 1 <= threshold <= 5:
        raise ValueError("Threshold must be between 1 and 5.")

    if user_id not in ratings["user_id"].values:
        raise ValueError("User ID not found.")

    user_ratings = ratings[ratings["user_id"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= threshold]
    genre_cols = [col for col in GENRE_COLS if col in movies.columns]
    merged = liked[["movie_id"]].merge(
        movies[["movie_id", *genre_cols]],
        on="movie_id",
        how="inner",
    )
    genre_counts = merged[genre_cols].sum().sort_values(ascending=False)
    genre_counts = genre_counts[genre_counts > 0].astype(int)

    return {
        "user_id": int(user_id),
        "total_ratings": int(len(user_ratings)),
        "positive_ratings": int(len(liked)),
        "mean_rating": float(user_ratings["rating"].mean()),
        "genre_counts": genre_counts,
    }


def compare_user_recommendations(models, ratings, movies, user_id, n=10, threshold=4):
    """
    Build a side-by-side table of top-n recommended titles across models.
    """
    title_map = movies.set_index("movie_id")["title"]
    columns = {}
    for name, model in models.items():
        recs = model.recommend(ratings, user_id=user_id, n=n, threshold=threshold)
        if "title" in recs.columns:
            titles = recs["title"].tolist()
        else:
            titles = recs["movie_id"].map(title_map).fillna("").tolist()
        if len(titles) < n:
            titles.extend([""] * (n - len(titles)))
        columns[name] = titles[:n]

    table = pd.DataFrame(columns, index=range(1, n + 1))
    table.index.name = "rank"
    return table
