import numpy as np
import pandas as pd


class RandomRecommender:
    """
    Floor baseline: recommends n random movies the user hasn't rated.

    Any real recommender should beat this. Results are reproducible: the
    same seed and user always get the same list.
    """

    def __init__(self, movies, seed=0):
        """
        Args:
            movies: DataFrame containing movie_id and title.
            seed: Random seed. Combined with user_id so each user gets
                a different (but repeatable) list.
        """
        self.movies = movies[["movie_id", "title"]].copy()
        self.seed = seed

    def recommend(self, ratings, user_id, n=10, threshold=4):
        """
        Recommend n random unrated movies.

        threshold is accepted (and ignored) so this has the same
        interface as the other recommenders.

        Returns:
            DataFrame containing movie_id, title.
        """
        if n <= 0:
            raise ValueError("n must be positive.")

        if user_id not in ratings["user_id"].values:
            raise ValueError("User ID not found.")

        rated_movies = ratings.loc[ratings["user_id"] == user_id, "movie_id"]
        candidates = self.movies[~self.movies["movie_id"].isin(rated_movies)]

        rng = np.random.default_rng([self.seed, int(user_id)])
        picks = rng.permutation(len(candidates))[:n]

        return candidates.iloc[picks].reset_index(drop=True)
