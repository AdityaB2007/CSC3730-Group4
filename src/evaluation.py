"""
Offline evaluation for comparing recommenders.

Each user's most recent ratings are held out as a test set. A recommender
only sees the training ratings, and we check how many of its top-k
recommendations are movies the user went on to rate positively.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.data import GENRE_COLS

PER_USER_METRICS = ("precision", "recall", "hit", "ndcg", "mrr")


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
        dcg = sum(discounts[i] for i, m in enumerate(recs) if m in relevant)
        idcg = float(discounts[:min(len(relevant), k)].sum())
        mrr = next(
            (1.0 / rank for rank, m in enumerate(recs, start=1) if m in relevant),
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


def _personalization(recommended_lists):
    lists = [list(dict.fromkeys(r)) for r in recommended_lists if len(r) > 0]
    n_users = len(lists)
    if n_users < 2:
        return 0.0

    item_cols = {}
    for recs in lists:
        for mid in recs:
            if mid not in item_cols:
                item_cols[mid] = len(item_cols)

    mat = np.zeros((n_users, len(item_cols)), dtype=float)
    for row, recs in enumerate(lists):
        mat[row, [item_cols[m] for m in recs]] = 1.0

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / norms
    sim = normed @ normed.T
    avg_sim = (sim.sum() - n_users) / (n_users * (n_users - 1))
    return float(np.clip(1.0 - avg_sim, 0.0, 1.0))


def compare_models(models, train, test, k=10, threshold=4, users=None):
    """
    Evaluate several recommenders on the same split.

    Args:
        models: Dict mapping a display name to a recommender.

    Returns:
        DataFrame with one row per model: precision@k, recall@k,
        hit_rate@k, ndcg@k, mrr@k, coverage, and personalization.
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
            "personalization": _personalization(results["recommended"]),
        })

    return pd.DataFrame(summary).set_index("model")


def evaluate_by_user_activity(models, train, test, k=10, threshold=4, quantiles=3):
    """
    Break down metrics across user training-activity buckets.
    """
    if quantiles < 1:
        raise ValueError("quantiles must be at least 1.")

    liked_users = set(test.loc[test["rating"] >= threshold, "user_id"])
    train_counts = train.groupby("user_id").size()
    users = sorted(u for u in liked_users if u in train_counts.index)

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
    if not users:
        return pd.DataFrame(columns=columns)

    counts = train_counts.loc[users]
    tier_ids = pd.qcut(counts, q=quantiles, labels=False, duplicates="drop")
    n_bins = int(tier_ids.nunique())

    if n_bins == 3:
        labels = ["Light", "Medium", "Heavy"]
    elif n_bins == 2:
        labels = ["Light", "Heavy"]
    elif n_bins == 1:
        labels = ["All"]
    else:
        labels = [f"Tier {i + 1}" for i in range(n_bins)]

    bins = sorted(tier_ids.unique())
    bin_map = {b: labels[i] for i, b in enumerate(bins)}
    user_segment = tier_ids.map(bin_map)

    rows = []
    for name, model in models.items():
        res = evaluate(model, train, test, k=k, threshold=threshold, users=users)
        res["segment"] = res["user_id"].map(user_segment)
        for seg in [bin_map[b] for b in bins]:
            sub = res[res["segment"] == seg]
            seg_counts = counts.loc[sub["user_id"]]
            rows.append({
                "segment": seg,
                "model": name,
                "users_evaluated": len(sub),
                "min_train_ratings": int(seg_counts.min()),
                "max_train_ratings": int(seg_counts.max()),
                f"precision@{k}": sub["precision"].mean(),
                f"recall@{k}": sub["recall"].mean(),
                f"hit_rate@{k}": sub["hit"].mean(),
                f"ndcg@{k}": sub["ndcg"].mean(),
                f"mrr@{k}": sub["mrr"].mean(),
            })

    return pd.DataFrame(rows, columns=columns)


def user_genre_profile(ratings, movies, user_id, threshold=4):
    """
    Return a summary of a user's rating counts and liked genres.
    """
    if not 1 <= threshold <= 5:
        raise ValueError("Threshold must be between 1 and 5.")

    if user_id not in ratings["user_id"].values:
        raise ValueError("User ID not found.")

    user_ratings = ratings[ratings["user_id"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= threshold]
    genre_cols = [c for c in GENRE_COLS if c in movies.columns]
    merged = liked[["movie_id"]].merge(
        movies[["movie_id", *genre_cols]], on="movie_id", how="inner"
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
    Show top-n recommended movie titles side by side across models.
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



# Significance testing
# - Paired: both models scored on the same users.
# - Wilcoxon: per-user scores are mostly zeros and skewed so we need not normal distribution tests
#  Holm: corrects p-values since we test 5 metrics at once


def bootstrap_ci(values, n_boot=1000, ci=0.95, random_state=13):
    """
    Bootstrap a confidence interval to the mean of per-user scores
    
    Args:
        values: One score per user.
        n_boot: Number of bootstrap resamples
        ci: Confidence level
        random_state: reproducible seed
        
        Returns:
            loats (mean, low, high)
        """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values must not be empty")
                         
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0-1")
    
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)

    alpha = (1 - ci) / 2
    low, high = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return float(values.mean()), float(low), float(high)


def _holm(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    scaled = (m - np.arange(m)) * p[order]

    adjusted = np.empty(m)
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def paired_significance(
        results_a, results_b, metrics=PER_USER_METRICS, n_boot=1000, ci=0.95, random_state=13
):
    """
    Test whether two recs differ, metric by metric
    
    Args:
        results_a: evaluate() output for model A
        results_b: evaluate() output for model B
        metrics: Per user metric columns to compare
        n_boot: Bootstrap resamples for the Ci
        ci: confidence level for the interval
        random_state: seed for bootstrap
        
    returns:
        Dataframe indexed by metric with mean_a, mean_b, mean_diff (A - B),
        ci_low, ci_high, a_better, b_better, ties, p_value, and p_holm
    """
    metrics = list(metrics)
    paired = results_a[["user_id", *metrics]].merge(
        results_b[["user_id", *metrics]],
        on="user_id",
        how="inner",
        suffixes=("_a", "_b"),
    )
    if paired.empty:
        raise ValueError("The two results share no users.")
    
    rows = []
    for metric in metrics:
            a = paired[f"{metric}_a"].astype(float).to_numpy()
            b = paired[f"{metric}_b"].astype(float).to_numpy()
            diff = a - b

            mean_diff, low, high = bootstrap_ci(
                diff, n_boot=n_boot, ci=ci, random_state=random_state        
            )
            if np.all(diff == 0):
                p_value = 1.0
            else:
                _, p_value = wilcoxon(diff)

            rows.append({
                "metric": metric,
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "mean_diff": mean_diff,
                "ci_low": low,
                "ci_high": high,
                "a_better": int((diff > 0).sum()),
                "b_better": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "p_value": p_value,
            })

    table = pd.DataFrame(rows).set_index("metric")
    table["p_holm"] = _holm(table["p_value"])
    return table