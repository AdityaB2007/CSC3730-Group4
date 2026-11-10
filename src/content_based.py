import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from src.data import GENRE_COLUMNS

class ContentBasedRecommender:
    """
    Genre-based content filtering using cosine similarity.
    """

    def __init__(self, movies):
        self.movies = movies.copy()

        self.genre_matrix = (
            self.movies
            .set_index("movie_id")[GENRE_COLUMNS]
        )

    def recommend(self, ratings, user_id, n=10, threshold=4):
        """
        Recommend movies based on a user's positively rated movies.

        Args:
            ratings: DataFrame containing user_id, movie_id, rating.
            user_id: User to generate recommendations for.
            n: Number of recommendations.
            threshold: Inclusive positive rating threshold.

        Returns:
            DataFrame containing movie_id, title, similarity.
        """

        if not 1 <= threshold <= 5:
            raise ValueError("Threshold must be between 1 and 5.")

        if n <= 0:
            raise ValueError("n must be positive.")

        if user_id not in ratings["user_id"].values:
            raise ValueError("User ID not found.")

        # Retrieve only this user's rating history.
        user_ratings = ratings[
            ratings["user_id"] == user_id
        ]

        rated_movies = user_ratings["movie_id"].unique()

        # Identify movies this user rated positively.
        positive_movies = user_ratings.loc[
            user_ratings["rating"] >= threshold,
            "movie_id"
        ].unique()

        if len(positive_movies) == 0:
            return pd.DataFrame(
                columns=["movie_id", "title", "similarity"]
            )

        # Identify movies the user has not rated.
        unrated_movies = self.genre_matrix.index.difference(
            rated_movies
        )

        if len(unrated_movies) == 0:
            return pd.DataFrame(
                columns=["movie_id", "title", "similarity"]
            )

        positive_vectors = self.genre_matrix.loc[
            positive_movies
        ]

        unrated_vectors = self.genre_matrix.loc[
            unrated_movies
        ]

        # Calculate similarity between each candidate and
        # every positively rated movie.
        similarity_matrix = cosine_similarity(
            unrated_vectors,
            positive_vectors
        )

        # Average similarities across positive movies.
        similarity_scores = similarity_matrix.mean(axis=1)

        recommendations = pd.DataFrame({
            "movie_id": unrated_movies,
            "similarity": similarity_scores
        })

        # Attach movie titles.
        recommendations = recommendations.merge(
            self.movies[["movie_id", "title"]],
            on="movie_id",
            how="left"
        )

        # Rank recommendations by similarity.
        recommendations = recommendations.sort_values(
            by="similarity",
            ascending=False
        )

        return recommendations[
            ["movie_id", "title", "similarity"]
        ].head(n).reset_index(drop=True)