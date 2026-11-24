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

        self._sim_array = self.similarity.to_numpy()
        self._movie_ids = self.genre_matrix.index.to_numpy()
        self._titles = (
            self.movies
            .set_index("movie_id")
            .loc[self._movie_ids, "title"]
            .to_numpy()
        )
        self._pos = {mid: i for i, mid in enumerate(self._movie_ids)}

        self._cache_ratings = None
        self._cache_counts = None
        self._cache_users = None

    def _update_cache(self, ratings):
        if ratings is self._cache_ratings:
            return

        counts = ratings["movie_id"].value_counts()
        self._cache_counts = (
            pd.Series(self._movie_ids).map(counts).fillna(0).to_numpy()
        )
        self._cache_users = {
            uid: (grp["movie_id"].to_numpy(), grp["rating"].to_numpy())
            for uid, grp in ratings.groupby("user_id", sort=False)
        }
        self._cache_ratings = ratings

    def _score_candidates(self, ratings, user_id, threshold=4):
        if not 1 <= threshold <= 5:
            raise ValueError("Threshold must be between 1 and 5.")

        self._update_cache(ratings)

        if user_id not in self._cache_users:
            raise ValueError("User ID not found.")

        rated_movies, user_ratings = self._cache_users[user_id]
        unrated_mask = ~np.isin(self._movie_ids, rated_movies)

        positive_movies = pd.unique(rated_movies[user_ratings >= threshold])
        pos_idx = [self._pos[m] for m in positive_movies if m in self._pos]

        if len(pos_idx) == 0 or not unrated_mask.any():
            empty_scores = np.zeros(int(unrated_mask.sum()), dtype=float)
            return unrated_mask, empty_scores, len(pos_idx) > 0

        scores = self._sim_array[:, pos_idx].mean(axis=1)[unrated_mask]
        return unrated_mask, scores, True

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

        unrated_mask, similarity_scores, has_positive = self._score_candidates(
            ratings, user_id, threshold=threshold
        )

        if not has_positive or not unrated_mask.any():
            return pd.DataFrame(
                columns=["movie_id", "title", "similarity"]
            )

        recommendations = pd.DataFrame({
            "movie_id": self._movie_ids[unrated_mask],
            "title": self._titles[unrated_mask],
            "similarity": similarity_scores,
            "num_ratings": self._cache_counts[unrated_mask],
        })

        # Many movies share the exact same genre vector, so they tie on
        # similarity. Break ties by how often each movie was rated (in the
        # ratings we were given), then by movie_id so results are
        # deterministic. Without this, ties fall back to movie_id alone and
        # the top-n is effectively arbitrary.
        recommendations = recommendations.sort_values(
            by=["similarity", "num_ratings", "movie_id"],
            ascending=[False, False, True]
        )

        return recommendations[
            ["movie_id", "title", "similarity"]
        ].head(n).reset_index(drop=True)