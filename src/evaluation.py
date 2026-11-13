"""
Offline evaluation for comparing recommenders.

Each user's most recent ratings are held out as a test set. A recommender
only sees the training ratings, and we check how many of its top-k
recommendations are movies the user went on to rate positively.
"""

import numpy as np
import pandas as pd


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
        and the list of recommended movie_ids.
    """
    liked = (
        test[test["rating"] >= threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
    )

    train_users = set(train["user_id"])
    if users is None:
        users = [u for u in liked.index if u in train_users]

    rows = []
    for user_id in users:
        relevant = liked.get(user_id, set())
        if not relevant or user_id not in train_users:
            continue

        recs = model.recommend(
            train, user_id=user_id, n=k, threshold=threshold
        )["movie_id"].tolist()

        hits = len(relevant.intersection(recs))
        rows.append({
            "user_id": user_id,
            "precision": hits / k,
            "recall": hits / len(relevant),
            "hit": hits > 0,
            "recommended": recs,
        })

    return pd.DataFrame(
        rows,
        columns=["user_id", "precision", "recall", "hit", "recommended"]
    )


def compare_models(models, train, test, k=10, threshold=4, users=None):
    """
    Evaluate several recommenders on the same split.

    Args:
        models: Dict mapping a display name to a recommender.

    Returns:
        DataFrame with one row per model: precision@k, recall@k,
        hit_rate@k, and catalog coverage (share of all movies that
        were recommended to at least one user).
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
            "coverage": len(recommended) / catalog_size,
        })

    return pd.DataFrame(summary).set_index("model")
