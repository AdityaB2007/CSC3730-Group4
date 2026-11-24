import pandas as pd


class PopularityRecommender:
    """
    Non-personalized baseline: recommends movies that many users have rated
    and that a high share of those users rated positively.

    Every user gets the same ranking, minus the movies they have already rated.
    """

    def __init__(self, movies, min_ratings=50):
        """
        Args:
            movies: DataFrame containing movie_id and title.
            min_ratings: A movie needs at least this many ratings to be
                recommended, so a film with a single 5-star rating can't
                come out on top.
        """
        if min_ratings < 1:
            raise ValueError("min_ratings must be at least 1.")

        self.movies = movies[["movie_id", "title"]].copy()
        self.min_ratings = min_ratings

        # The ranking is the same for every user, so remember the last one.
        # Evaluation calls recommend() once per user with the same training
        # DataFrame; without this the ranking is rebuilt ~900 times.
        self._cache_ratings = None
        self._cache_threshold = None
        self._cache_result = None

    def rank_movies(self, ratings, threshold=4):
        """
        Score every movie with at least min_ratings ratings.

        The result is cached for the most recent (ratings, threshold) pair.
        The cache is keyed on the DataFrame object itself, so don't modify a
        ratings DataFrame in place between calls; pass a new one instead.

        Returns:
            DataFrame containing movie_id, title, num_ratings,
            positive_ratio, sorted best first.
        """
        if not 1 <= threshold <= 5:
            raise ValueError("Threshold must be between 1 and 5.")

        if ratings is self._cache_ratings and threshold == self._cache_threshold:
            return self._cache_result.copy()

        result = self._rank(ratings, threshold)
        self._cache_ratings = ratings
        self._cache_threshold = threshold
        self._cache_result = result
        return result.copy()

    def _rank(self, ratings, threshold):
        stats = (
            ratings
            .assign(positive=ratings["rating"] >= threshold)
            .groupby("movie_id")
            .agg(
                num_ratings=("rating", "size"),
                positive_ratio=("positive", "mean")
            )
            .reset_index()
        )

        stats = stats[stats["num_ratings"] >= self.min_ratings]

        # Highest share of positive ratings first; break ties
        # with the number of ratings, then movie_id for determinism.
        stats = stats.sort_values(
            by=["positive_ratio", "num_ratings", "movie_id"],
            ascending=[False, False, True]
        )

        return stats.merge(
            self.movies, on="movie_id", how="left"
        )[["movie_id", "title", "num_ratings", "positive_ratio"]]

    def recommend(self, ratings, user_id, n=10, threshold=4):
        """
        Recommend the most popular movies the user has not rated yet.

        Args:
            ratings: DataFrame containing user_id, movie_id, rating.
            user_id: User to generate recommendations for.
            n: Number of recommendations.
            threshold: Inclusive positive rating threshold.

        Returns:
            DataFrame containing movie_id, title, num_ratings, positive_ratio.
        """
        if n <= 0:
            raise ValueError("n must be positive.")

        if user_id not in ratings["user_id"].values:
            raise ValueError("User ID not found.")

        ranked = self.rank_movies(ratings, threshold=threshold)

        rated_movies = ratings.loc[
            ratings["user_id"] == user_id, "movie_id"
        ]

        return (
            ranked[~ranked["movie_id"].isin(rated_movies)]
            .head(n)
            .reset_index(drop=True)
        )
