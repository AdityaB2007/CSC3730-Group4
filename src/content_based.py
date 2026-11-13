import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from src.data import GENRE_COLS

class ContentBasedRecommender:
    """
    Genre-based content filtering using cosine similarity.
    """

    def __init__(self, movies):
        self.movies = movies.copy()

        self.genre_matrix = (
            self.movies
            .set_index("movie_id")[GENRE_COLS]
        )

        # Movie-to-movie similarity only depends on genres, so compute it
        # once here instead of on every recommend() call. This matters when
        # the evaluation asks for recommendations for every user.
        self.similarity = pd.DataFrame(
            cosine_similarity(self.genre_matrix),
            index=self.genre_matrix.index,
            columns=self.genre_matrix.index
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

        # Look up similarity between each candidate and
        # every positively rated movie.
        similarity_matrix = self.similarity.loc[
            unrated_movies,
            positive_movies
        ].to_numpy()

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
        # Stable sort so ties always come back in the same order.
        recommendations = recommendations.sort_values(
            by="similarity",
            ascending=False,
            kind="stable"
        )

        return recommendations[
            ["movie_id", "title", "similarity"]
        ].head(n).reset_index(drop=True)