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
    model = ContentBasedRecommender(sample_movies)
    results = model.recommend(no_positive, user_id=1, n=10)
    assert results.empty
    assert list(results.columns) == ["movie_id", "title", "similarity"]

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

def test_ties_broken_by_number_of_ratings(sample_movies):
    # Movies 10 and 30 are both pure Action, so they tie on similarity for
    # a user who liked a third Action movie. The more-rated one should win,
    # even though it has the larger movie_id.
    movies = pd.concat([
        sample_movies,
        sample_movies.iloc[[0]].assign(movie_id=50, title="Action Three")
    ], ignore_index=True)
    ratings = pd.DataFrame({
        "user_id":  [1, 2, 3, 2],
        "movie_id": [50, 30, 30, 10],
        "rating":   [5, 4, 4, 4],
    })
    model = ContentBasedRecommender(movies)
    results = model.recommend(ratings, user_id=1, n=2)
    assert results["movie_id"].tolist() == [30, 10]


def test_zero_genre_vector_handled_safely(sample_movies):
    zero_movie = sample_movies.iloc[[0]].assign(movie_id=50, title="No Genres", Action=0)
    movies = pd.concat([sample_movies, zero_movie], ignore_index=True)
    ratings = pd.DataFrame({
        "user_id": [1, 2],
        "movie_id": [10, 50],
        "rating": [5, 5],
    })
    model = ContentBasedRecommender(movies)
    results = model.recommend(ratings, user_id=1, n=10)
    zero_row = results[results["movie_id"] == 50]
    assert len(zero_row) == 1
    assert zero_row.iloc[0]["similarity"] == pytest.approx(0.0)
    assert not results["similarity"].isna().any()

    # User who only liked the zero-genre movie still gets valid non-NaN output.
    from_zero = model.recommend(ratings, user_id=2, n=10)
    assert (from_zero["similarity"] == 0.0).all()


def test_all_movies_rated_returns_empty(sample_movies):
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "movie_id": [10, 20, 30, 40],
        "rating": [5, 4, 5, 4],
    })
    model = ContentBasedRecommender(sample_movies)
    results = model.recommend(ratings, user_id=1, n=10)
    assert results.empty
    assert list(results.columns) == ["movie_id", "title", "similarity"]


def test_unknown_movie_id_in_ratings_ignored(sample_movies):
    ratings = pd.DataFrame({
        "user_id": [1, 2, 2],
        "movie_id": [999, 999, 10],
        "rating": [5, 5, 5],
    })
    model = ContentBasedRecommender(sample_movies)
    only_unknown = model.recommend(ratings, user_id=1, n=10)
    assert only_unknown.empty
    assert list(only_unknown.columns) == ["movie_id", "title", "similarity"]

    mixed = model.recommend(ratings, user_id=2, n=10)
    assert mixed.iloc[0]["movie_id"] == 30


def test_user_profiles_few_concentrated_and_diverse_ratings(sample_movies):
    movies = sample_movies.copy()
    movies.loc[2, "Sci-Fi"] = 1  # movie 30 is Action+Sci-Fi
    ratings = pd.DataFrame({
        "user_id":  [1, 2, 2, 3, 3],
        "movie_id": [10, 10, 30, 10, 20],
        "rating":   [5, 5, 5, 5, 5],
    })
    model = ContentBasedRecommender(movies)
    for uid, rated in [(1, {10}), (2, {10, 30}), (3, {10, 20})]:
        res = model.recommend(ratings, user_id=uid, n=100)
        assert res["movie_id"].is_unique
        assert not set(res["movie_id"]) & rated
        assert len(res) == len(movies) - len(rated)


def test_cache_recomputes_for_new_ratings_dataframe(sample_movies):
    movies = pd.concat([
        sample_movies,
        sample_movies.iloc[[0]].assign(movie_id=50, title="Action Three")
    ], ignore_index=True)
    ratings_a = pd.DataFrame({
        "user_id":  [1, 2, 3, 2],
        "movie_id": [50, 30, 30, 10],
        "rating":   [5, 4, 4, 4],
    })
    ratings_b = pd.concat([
        ratings_a,
        pd.DataFrame({"user_id": [4, 5], "movie_id": [10, 10], "rating": [4, 4]})
    ], ignore_index=True)

    model = ContentBasedRecommender(movies)
    assert model.recommend(ratings_a, user_id=1, n=2)["movie_id"].tolist() == [30, 10]
    assert model.recommend(ratings_b, user_id=1, n=2)["movie_id"].tolist() == [10, 30]
