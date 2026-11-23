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
        sim_values = cosine_similarity(self.genre_matrix)
        self.similarity = pd.DataFrame(
            sim_values,
            index=self.genre_matrix.index,
            columns=self.genre_matrix.index
        )

        # Precompute aligned arrays for fast per-user candidate lookup.
        self._sim_values = sim_values
        self._movie_ids = self.genre_matrix.index.to_numpy()
        self._titles = (
            self.movies
            .set_index("movie_id")
            .loc[self._movie_ids, "title"]
            .to_numpy()
        )
        self._id_to_pos = {mid: idx for idx, mid in enumerate(self._movie_ids)}

        # Per-ratings cache keyed by DataFrame object identity (same contract
        # as PopularityRecommender: pass a new DataFrame rather than mutating).
        self._cache_ratings = None
        self._cache_counts = None
        self._cache_user_data = None

    def _ensure_cache(self, ratings):
        if ratings is self._cache_ratings:
            return

        counts = ratings["movie_id"].value_counts()
        aligned_counts = (
            pd.Series(self._movie_ids)
            .map(counts)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

        user_data = {}
        if len(ratings) > 0:
            u_arr = ratings["user_id"].to_numpy()
            m_arr = ratings["movie_id"].to_numpy()
            r_arr = ratings["rating"].to_numpy()
            order = np.argsort(u_arr, kind="stable")
            su, sm, sr = u_arr[order], m_arr[order], r_arr[order]
            uniq_u, starts = np.unique(su, return_index=True)
            ends = np.append(starts[1:], len(su))
            for u, start, end in zip(uniq_u, starts, ends):
                key = u.item() if hasattr(u, "item") else u
                user_data[key] = (sm[start:end], sr[start:end])

        self._cache_ratings = ratings
        self._cache_counts = aligned_counts
        self._cache_user_data = user_data

    def _score_unrated(self, ratings, user_id, threshold=4):
        """
        Compute unrated candidate mask, similarity scores, and positive-rating flag.
        """
        if not 1 <= threshold <= 5:
            raise ValueError("Threshold must be between 1 and 5.")

        self._ensure_cache(ratings)

        user_entry = self._cache_user_data.get(user_id)
        if user_entry is None:
            raise ValueError("User ID not found.")

        rated_movies, user_ratings = user_entry
        unrated_mask = ~np.isin(self._movie_ids, rated_movies)

        pos_raw = rated_movies[user_ratings >= threshold]
        pos_idx = [
            self._id_to_pos[m]
            for m in dict.fromkeys(pos_raw)
            if m in self._id_to_pos
        ]

        if len(pos_idx) == 0 or not unrated_mask.any():
            return unrated_mask, np.zeros(int(unrated_mask.sum()), dtype=float), len(pos_idx) > 0

        similarity_scores = self._sim_values[:, pos_idx].mean(axis=1)[unrated_mask]
        return unrated_mask, similarity_scores, True

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

        unrated_mask, similarity_scores, has_positive = self._score_unrated(
            ratings, user_id, threshold=threshold
        )

        if not has_positive or not unrated_mask.any():
            return pd.DataFrame(
                columns=["movie_id", "title", "similarity"]
            )

        cand_ids = self._movie_ids[unrated_mask]
        cand_titles = self._titles[unrated_mask]
        cand_counts = self._cache_counts[unrated_mask]

        # Many movies share the exact same genre vector, so they tie on
        # similarity. Break ties by how often each movie was rated (in the
        # ratings we were given), then by movie_id so results are deterministic.
        top_order = np.lexsort((cand_ids, -cand_counts, -similarity_scores))[:n]

        return pd.DataFrame({
            "movie_id": cand_ids[top_order],
            "title": cand_titles[top_order],
            "similarity": similarity_scores[top_order],
        })