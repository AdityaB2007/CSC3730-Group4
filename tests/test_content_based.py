import pandas as pd
import pytest

from src.data import GENRE_COLS
from src.content_based import ContentBasedRecommender

@pytest.fixture
def sample_movies():
    movies = pd.DataFrame(
        0,
        index=range(4),
        columns=GENRE_COLS
    )

    movies.insert(0, "title", [
        "Action Movie",
        "Comedy Movie",
        "Action Sequel",
        "Comedy Sequel"
    ])

    movies.insert(0, "movie_id", [10, 20, 30, 40])
    movies.loc[0, "Action"] = 1
    movies.loc[1, "Comedy"] = 1
    movies.loc[2, "Action"] = 1
    movies.loc[3, "Comedy"] = 1
    return movies

@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "movie_id": [10, 20, 10, 30],
        "rating": [5, 2, 2, 5]
    })

def test_recommendations_exclude_rated_movies(sample_movies, sample_ratings):
    model = ContentBasedRecommender(sample_movies)
    results = model.recommend(sample_ratings, user_id=1, n=10)
    assert 10 not in results["movie_id"].values
    assert 20 not in results["movie_id"].values

def test_similar_genres_receive_higher_scores(sample_movies, sample_ratings):
    model = ContentBasedRecommender(sample_movies)
    results = model.recommend(sample_ratings, user_id=1, n=10)
    assert results.iloc[0]["movie_id"] == 30

def test_no_positive_ratings(sample_movies, sample_ratings):
    no_positive = pd.DataFrame({
        "user_id": [1, 1],
        "movie_id": [10, 20],
        "rating": [2, 3]
    })

def test_invalid_user(sample_movies, sample_ratings):
    model = ContentBasedRecommender(sample_movies)
    with pytest.raises(ValueError):
        model.recommend(sample_ratings, user_id=999)

def test_invalid_threshold(sample_movies, sample_ratings):
    model = ContentBasedRecommender(sample_movies)
    with pytest.raises(ValueError):
        model.recommend(sample_ratings, user_id=1, threshold=6)

def test_invalid_recommendation_count(sample_movies, sample_ratings):
    model = ContentBasedRecommender(sample_movies)
    with pytest.raises(ValueError):
        model.recommend(sample_ratings, user_id=1, n=0)

def test_recommendation_scores_are_valid(sample_movies, sample_ratings):
    model = ContentBasedRecommender(sample_movies)
    results = model.recommend(sample_ratings, user_id=1, n=10)
    assert results["similarity"].between(0, 1).all()
    assert results["similarity"].is_monotonic_decreasing