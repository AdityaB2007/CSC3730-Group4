import pandas as pd
import pytest

from src.evaluation import compare_models, evaluate, train_test_split_by_time


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
