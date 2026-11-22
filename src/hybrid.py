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

        if user_id not in ratings["user_id"].values:
            raise ValueError("User ID not found.")

        rated_movies = ratings.loc[
            ratings["user_id"] == user_id, "movie_id"
        ]

        candidates = self.movies[
            ~self.movies["movie_id"].isin(rated_movies)
        ]

        # Score every candidate, not just each model's top n.
        content_scores = self.content.recommend(
            ratings, user_id, n=len(self.movies), threshold=threshold
        ).set_index("movie_id")["similarity"]

        popularity_scores = self.popularity.rank_movies(
            ratings, threshold=threshold
        ).set_index("movie_id")["positive_ratio"]

        # Movies with no score (no positive ratings for content,
        # too few ratings for popularity) get 0.
        recommendations = candidates.assign(
            content_score=candidates["movie_id"].map(content_scores).fillna(0.0),
            popularity_score=candidates["movie_id"].map(popularity_scores).fillna(0.0)
        )

        # Rescale both to 0-1 so alpha means the same thing for every user.
        for col in ["content_score", "popularity_score"]:
            recommendations[col] = min_max_scale(recommendations[col])

        recommendations["score"] = (
            self.alpha * recommendations["content_score"]
            + (1 - self.alpha) * recommendations["popularity_score"]
        )

        recommendations = recommendations.sort_values(
            by=["score", "movie_id"],
            ascending=[False, True]
        )

        return recommendations[
            ["movie_id", "title", "content_score", "popularity_score", "score"]
        ].head(n).reset_index(drop=True)


def min_max_scale(values):
    low, high = values.min(), values.max()

    if high == low:
        return pd.Series(0.0, index=values.index)

    return (values - low) / (high - low)
