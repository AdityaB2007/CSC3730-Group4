import pandas as pd
import pytest

from src.data import GENRE_COLS
from src.hybrid import HybridRecommender


@pytest.fixture
def movies():
    movies = pd.DataFrame(0, index=range(4), columns=GENRE_COLS)
    movies.insert(0, "title", ["Action A", "Comedy A", "Action B", "Comedy B"])
    movies.insert(0, "movie_id", [1, 2, 3, 4])
    movies.loc[[0, 2], "Action"] = 1
    movies.loc[[1, 3], "Comedy"] = 1
    return movies


@pytest.fixture
def ratings():
    # User 1 likes action (movie 1) and hasn't seen 3 or 4.
    # Comedy B (4) is loved by everyone else; Action B (3) is disliked.
    return pd.DataFrame({
        "user_id":  [1, 1, 2, 2, 2, 3, 3, 3],
        "movie_id": [1, 2, 3, 4, 1, 3, 4, 2],
        "rating":   [5, 2, 1, 5, 4, 2, 5, 3],
    })


def test_alpha_one_follows_genres(movies, ratings):
    model = HybridRecommender(movies, alpha=1, min_ratings=1)
    results = model.recommend(ratings, user_id=1)
    assert results.iloc[0]["movie_id"] == 3


def test_alpha_zero_follows_popularity(movies, ratings):
    model = HybridRecommender(movies, alpha=0, min_ratings=1)
    results = model.recommend(ratings, user_id=1)
    assert results.iloc[0]["movie_id"] == 4


def test_excludes_rated_movies(movies, ratings):
    model = HybridRecommender(movies, alpha=0.5, min_ratings=1)
    results = model.recommend(ratings, user_id=1)
    assert set(results["movie_id"]) == {3, 4}


def test_scores_are_scaled(movies, ratings):
    model = HybridRecommender(movies, alpha=0.5, min_ratings=1)
    results = model.recommend(ratings, user_id=1)
    for col in ["content_score", "popularity_score", "score"]:
        assert results[col].between(0, 1).all()
    assert results["score"].is_monotonic_decreasing


def test_user_without_positive_ratings_gets_popular_movies(movies, ratings):
    no_positive = ratings[ratings["user_id"] != 1]
    no_positive = pd.concat([
        no_positive,
        pd.DataFrame({"user_id": [1], "movie_id": [1], "rating": [1]})
    ])
    model = HybridRecommender(movies, alpha=0.5, min_ratings=1)
    results = model.recommend(no_positive, user_id=1)
    assert results.iloc[0]["movie_id"] == 4


def test_invalid_inputs(movies, ratings):
    with pytest.raises(ValueError):
        HybridRecommender(movies, alpha=1.5)

    model = HybridRecommender(movies)
    with pytest.raises(ValueError):
        model.recommend(ratings, user_id=999)
    with pytest.raises(ValueError):
        model.recommend(ratings, user_id=1, n=0)
    with pytest.raises(ValueError):
        model.recommend(ratings, user_id=1, threshold=0)
