import numpy as np
import pandas as pd

from src.content_based import ContentBasedRecommender
from src.popularity import PopularityRecommender


class HybridRecommender:
    """
    Weighted blend of content-based and popularity scores.

    score = alpha * content_score + (1 - alpha) * popularity_score
    """

    def __init__(self, movies, alpha=0.5, min_ratings=50):
        """
        Args:
            movies: DataFrame containing movie_id, title, and genre columns.
            alpha: Weight on the content-based score (0 to 1).
                alpha=1 is pure content-based, alpha=0 is pure popularity.
            min_ratings: Passed to PopularityRecommender.
        """

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        self.movies = movies[["movie_id", "title"]].copy()
        self.alpha = alpha
        self.content = ContentBasedRecommender(movies)
        self.popularity = PopularityRecommender(movies, min_ratings=min_ratings)

        self._cache_ratings = None
        self._cache_threshold = None
        self._cache_pop_scores = None

    def _popularity_scores(self, ratings, threshold):
        if ratings is self._cache_ratings and threshold == self._cache_threshold:
            return self._cache_pop_scores

        ranked = self.popularity.rank_movies(ratings, threshold=threshold)
        pop_map = ranked.set_index("movie_id")["positive_ratio"]
        scores = (
            pd.Series(self.content._movie_ids)
            .map(pop_map)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        self._cache_ratings = ratings
        self._cache_threshold = threshold
        self._cache_pop_scores = scores
        return scores

    def recommend(self, ratings, user_id, n=10, threshold=4):
        """
        Recommend unrated movies ranked by the blended score.

        Args:
            ratings: DataFrame containing user_id, movie_id, rating.
            user_id: User to generate recommendations for.
            n: Number of recommendations.
            threshold: Inclusive positive rating threshold.

        Returns:
            DataFrame containing movie_id, title, content_score,
            popularity_score, score.
        """

        if not 1 <= threshold <= 5:
            raise ValueError("Threshold must be between 1 and 5.")

        if n <= 0:
            raise ValueError("n must be positive.")

        unrated_mask, raw_content, _ = self.content._score_candidates(
            ratings, user_id, threshold=threshold
        )

        if not unrated_mask.any():
            return pd.DataFrame(
                columns=["movie_id", "title", "content_score", "popularity_score", "score"]
            )

        raw_popularity = self._popularity_scores(ratings, threshold)[unrated_mask]

        # Rescale both to 0-1 so alpha means the same thing for every user.
        recommendations = pd.DataFrame({
            "movie_id": self.content._movie_ids[unrated_mask],
            "title": self.content._titles[unrated_mask],
            "content_score": min_max_scale(raw_content),
            "popularity_score": min_max_scale(raw_popularity),
            "num_ratings": self.content._cache_counts[unrated_mask],
        })

        recommendations["score"] = (
            self.alpha * recommendations["content_score"]
            + (1 - self.alpha) * recommendations["popularity_score"]
        )

        # Break score ties by how often a movie was rated, then movie_id,
        # matching ContentBasedRecommender.
        recommendations = recommendations.sort_values(
            by=["score", "num_ratings", "movie_id"],
            ascending=[False, False, True]
        )

        return recommendations[
            ["movie_id", "title", "content_score", "popularity_score", "score"]
        ].head(n).reset_index(drop=True)


def min_max_scale(values):
    if len(values) == 0:
        return pd.Series(dtype=float) if isinstance(values, pd.Series) else np.array([], dtype=float)

    low, high = values.min(), values.max()

    if high == low:
        if isinstance(values, pd.Series):
            return pd.Series(0.0, index=values.index)
        return np.zeros(len(values), dtype=float)

    return (values - low) / (high - low)
