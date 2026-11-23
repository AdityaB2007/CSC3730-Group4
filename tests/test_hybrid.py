import numpy as np
import pandas as pd
import pytest

from src.data import GENRE_COLS
from src.hybrid import HybridRecommender, min_max_scale


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


def test_ties_broken_by_number_of_ratings(movies):
    extra = movies.iloc[[0]].assign(movie_id=5, title="Action C")
    all_movies = pd.concat([movies, extra], ignore_index=True)
    # User 1 liked movie 1 (Action). Both 3 and 5 are Action and equally liked.
    # Movie 5 has more ratings than movie 3, so it should win the tie.
    ratings = pd.DataFrame({
        "user_id":  [1, 2, 3, 2],
        "movie_id": [1, 5, 5, 3],
        "rating":   [5, 5, 5, 5],
    })
    model = HybridRecommender(all_movies, alpha=0.5, min_ratings=1)
    results = model.recommend(ratings, user_id=1, n=2)
    assert results["movie_id"].tolist() == [5, 3]


def test_all_movies_rated_returns_empty(movies):
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "movie_id": [1, 2, 3, 4],
        "rating": [5, 4, 3, 2],
    })
    model = HybridRecommender(movies, alpha=0.5, min_ratings=1)
    results = model.recommend(ratings, user_id=1, n=10)
    assert results.empty
    assert list(results.columns) == [
        "movie_id", "title", "content_score", "popularity_score", "score"
    ]


def test_hybrid_cache_updates_for_new_ratings(movies, ratings):
    model = HybridRecommender(movies, alpha=0.0, min_ratings=1)
    first = model.recommend(ratings, user_id=1)
    assert first.iloc[0]["movie_id"] == 4

    # In a new ratings frame where Action B (3) dominates Comedy B (4),
    # alpha=0.0 popularity ranking must flip.
    flipped = pd.DataFrame({
        "user_id":  [1, 2, 3, 2, 3],
        "movie_id": [1, 3, 3, 4, 4],
        "rating":   [5, 5, 5, 1, 1],
    })
    second = model.recommend(flipped, user_id=1)
    assert second.iloc[0]["movie_id"] == 3


def test_min_max_scale_edge_cases():
    empty_series = pd.Series(dtype=float)
    assert min_max_scale(empty_series).empty

    const_series = pd.Series([3.0, 3.0, 3.0], index=[10, 20, 30])
    scaled_series = min_max_scale(const_series)
    assert (scaled_series == 0.0).all()
    assert list(scaled_series.index) == [10, 20, 30]

    assert len(min_max_scale(np.array([], dtype=float))) == 0
    assert np.allclose(min_max_scale(np.array([2.5, 2.5])), [0.0, 0.0])
