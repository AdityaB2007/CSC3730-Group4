import numpy as np
import pandas as pd
import pytest

from src.data import GENRE_COLS
from src.evaluation import (
    _personalization,
    compare_models,
    compare_user_recommendations,
    evaluate,
    evaluate_by_user_activity,
    train_test_split_by_time,
    user_genre_profile,
)


@pytest.fixture
def ratings():
    # User 1 has 5 ratings, user 2 has 10, user 3 has a single rating.
    rows = []
    for t, movie in enumerate([1, 2, 3, 4, 5]):
        rows.append((1, movie, 5, t))
    for t, movie in enumerate(range(1, 11)):
        rows.append((2, movie, 4 if movie % 2 else 2, t))
    rows.append((3, 1, 5, 0))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


class FixedRecommender:
    """Always recommends the same list, for predictable metrics."""

    def __init__(self, movie_ids):
        self.movie_ids = movie_ids

    def recommend(self, ratings, user_id, n=10, threshold=4):
        return pd.DataFrame({"movie_id": self.movie_ids[:n]})


def test_split_holds_out_most_recent(ratings):
    train, test = train_test_split_by_time(ratings, test_fraction=0.2)
    assert test.loc[test["user_id"] == 1, "movie_id"].tolist() == [5]
    assert test.loc[test["user_id"] == 2, "movie_id"].tolist() == [9, 10]


def test_split_keeps_every_rating_once(ratings):
    train, test = train_test_split_by_time(ratings)
    assert len(train) + len(test) == len(ratings)
    assert pd.concat([train, test]).duplicated(["user_id", "movie_id"]).sum() == 0


def test_split_keeps_single_rating_users_in_train(ratings):
    train, test = train_test_split_by_time(ratings)
    assert 3 in train["user_id"].values
    assert 3 not in test["user_id"].values


def test_split_rejects_bad_fraction(ratings):
    with pytest.raises(ValueError):
        train_test_split_by_time(ratings, test_fraction=1.5)


def test_metrics(ratings):
    train, test = train_test_split_by_time(ratings, test_fraction=0.2)
    # User 1 liked movie 5 in test; user 2 liked movie 9 (10 is a 2).
    model = FixedRecommender([5, 7, 8])
    results = evaluate(model, train, test, k=2).set_index("user_id")

    assert results.loc[1, "precision"] == pytest.approx(0.5)
    assert results.loc[1, "recall"] == pytest.approx(1.0)
    assert results.loc[2, "precision"] == pytest.approx(0.0)
    assert not results.loc[2, "hit"]


def test_compare_models(ratings):
    train, test = train_test_split_by_time(ratings, test_fraction=0.2)
    summary = compare_models(
        {"good": FixedRecommender([5, 9]), "bad": FixedRecommender([7, 8])},
        train, test, k=2
    )
    assert summary.loc["good", "hit_rate@2"] == pytest.approx(1.0)
    assert summary.loc["bad", "hit_rate@2"] == pytest.approx(0.0)
    assert summary.loc["good", "coverage"] == pytest.approx(2 / 10)
    assert summary.loc["good", "ndcg@2"] > summary.loc["bad", "ndcg@2"]
    assert summary.loc["good", "mrr@2"] > summary.loc["bad", "mrr@2"]
    assert summary.loc["good", "personalization"] == pytest.approx(0.0)


def test_ndcg_and_mrr(ratings):
    train, test = train_test_split_by_time(ratings, test_fraction=0.2)
    model = FixedRecommender([7, 5, 9])
    results = evaluate(model, train, test, k=3).set_index("user_id")

    assert results.loc[1, "mrr"] == pytest.approx(0.5)
    assert results.loc[1, "ndcg"] == pytest.approx(1.0 / np.log2(3))
    assert results.loc[2, "mrr"] == pytest.approx(1.0 / 3.0)
    assert results.loc[2, "ndcg"] == pytest.approx(1.0 / np.log2(4))

    miss = evaluate(FixedRecommender([7, 8]), train, test, k=2).set_index("user_id")
    assert (miss["ndcg"] == 0.0).all()
    assert (miss["mrr"] == 0.0).all()


def test_personalization():
    assert _personalization([[1, 2], [1, 2], [1, 2]]) == pytest.approx(0.0)
    assert _personalization([[1, 2], [3, 4], [5, 6]]) == pytest.approx(1.0)
    assert _personalization([[1, 2]]) == pytest.approx(0.0)
    assert _personalization([[], [1, 2]]) == pytest.approx(0.0)


def test_evaluate_invalid_inputs(ratings):
    train, test = train_test_split_by_time(ratings, test_fraction=0.2)
    model = FixedRecommender([5])
    with pytest.raises(ValueError):
        evaluate(model, train, test, k=0)
    with pytest.raises(ValueError):
        evaluate(model, train, test, threshold=6)


def test_evaluate_by_user_activity(ratings):
    train, test = train_test_split_by_time(ratings, test_fraction=0.2)
    breakdown = evaluate_by_user_activity(
        {"fixed": FixedRecommender([5, 9])}, train, test, k=2, quantiles=2
    )
    assert set(breakdown["segment"]) == {"Light", "Heavy"}
    light = breakdown[breakdown["segment"] == "Light"].iloc[0]
    heavy = breakdown[breakdown["segment"] == "Heavy"].iloc[0]
    assert light["min_train_ratings"] == 4
    assert heavy["min_train_ratings"] == 8
    assert light["hit_rate@2"] == pytest.approx(1.0)
    assert heavy["hit_rate@2"] == pytest.approx(1.0)

    with pytest.raises(ValueError):
        evaluate_by_user_activity({"fixed": FixedRecommender([5])}, train, test, quantiles=0)


def test_user_profile_and_comparison():
    movies = pd.DataFrame(0, index=range(3), columns=GENRE_COLS)
    movies.insert(0, "title", ["Alpha", "Beta", "Gamma"])
    movies.insert(0, "movie_id", [1, 2, 3])
    movies.loc[[0, 1], "Action"] = 1
    movies.loc[0, "Sci-Fi"] = 1
    movies.loc[2, "Drama"] = 1

    ratings = pd.DataFrame({
        "user_id": [1, 1, 1],
        "movie_id": [1, 2, 3],
        "rating": [5, 4, 2],
    })

    profile = user_genre_profile(ratings, movies, user_id=1, threshold=4)
    assert profile["total_ratings"] == 3
    assert profile["positive_ratings"] == 2
    assert profile["genre_counts"]["Action"] == 2
    assert profile["genre_counts"]["Sci-Fi"] == 1
    assert "Drama" not in profile["genre_counts"]

    with pytest.raises(ValueError):
        user_genre_profile(ratings, movies, user_id=999)

    table = compare_user_recommendations(
        {"Model A": FixedRecommender([1, 2]), "Model B": FixedRecommender([3])},
        ratings,
        movies,
        user_id=1,
        n=3,
    )
    assert list(table.index) == [1, 2, 3]
    assert table["Model A"].tolist() == ["Alpha", "Beta", ""]
    assert table["Model B"].tolist() == ["Gamma", "", ""]
