import pandas as pd
import pytest

from src.random_baseline import RandomRecommender


@pytest.fixture
def movies():
    return pd.DataFrame({
        "movie_id": range(1, 21),
        "title": [f"Movie {i}" for i in range(1, 21)],
    })


@pytest.fixture
def ratings():
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "movie_id": [1, 2, 3, 4],
        "rating": [5, 4, 1, 3],
    })


def test_excludes_rated_and_returns_n(movies, ratings):
    results = RandomRecommender(movies).recommend(ratings, user_id=1, n=5)
    assert len(results) == 5
    assert results["movie_id"].is_unique
    assert not set(results["movie_id"]) & {1, 2, 3}


def test_reproducible_per_seed_and_user(movies, ratings):
    a = RandomRecommender(movies, seed=1).recommend(ratings, user_id=1)
    b = RandomRecommender(movies, seed=1).recommend(ratings, user_id=1)
    c = RandomRecommender(movies, seed=2).recommend(ratings, user_id=1)
    assert a["movie_id"].tolist() == b["movie_id"].tolist()
    assert a["movie_id"].tolist() != c["movie_id"].tolist()


def test_invalid_inputs(movies, ratings):
    model = RandomRecommender(movies)
    with pytest.raises(ValueError):
        model.recommend(ratings, user_id=999)
    with pytest.raises(ValueError):
        model.recommend(ratings, user_id=1, n=0)
