import pandas as pd
import pytest

from src.popularity import PopularityRecommender


@pytest.fixture
def movies():
    return pd.DataFrame({
        "movie_id": [1, 2, 3, 4],
        "title": ["Loved", "Mixed", "Niche Gem", "Disliked"],
    })


@pytest.fixture
def ratings():
    # Movie 1: 3 ratings, all positive.
    # Movie 2: 3 ratings, 1 positive.
    # Movie 3: 1 rating, positive (too few ratings to qualify).
    # Movie 4: 3 ratings, none positive.
    return pd.DataFrame({
        "user_id":  [1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 4],
        "movie_id": [1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4],
        "rating":   [5, 4, 5, 5, 2, 3, 5, 1, 2, 1, 2],
    })


def test_ranks_by_positive_ratio(movies, ratings):
    model = PopularityRecommender(movies, min_ratings=2)
    ranked = model.rank_movies(ratings)
    assert ranked["movie_id"].tolist() == [1, 2, 4]


def test_min_ratings_filters_rare_movies(movies, ratings):
    model = PopularityRecommender(movies, min_ratings=2)
    assert 3 not in model.rank_movies(ratings)["movie_id"].values


def test_excludes_movies_user_already_rated(movies, ratings):
    model = PopularityRecommender(movies, min_ratings=2)
    results = model.recommend(ratings, user_id=4, n=10)
    assert 4 not in results["movie_id"].values
    assert results["movie_id"].tolist() == [1, 2]


def test_respects_n(movies, ratings):
    model = PopularityRecommender(movies, min_ratings=1)
    assert len(model.recommend(ratings, user_id=4, n=1)) == 1


def test_invalid_inputs(movies, ratings):
    model = PopularityRecommender(movies)
    with pytest.raises(ValueError):
        model.recommend(ratings, user_id=999)
    with pytest.raises(ValueError):
        model.recommend(ratings, user_id=1, n=0)
    with pytest.raises(ValueError):
        model.recommend(ratings, user_id=1, threshold=6)
    with pytest.raises(ValueError):
        PopularityRecommender(movies, min_ratings=0)
