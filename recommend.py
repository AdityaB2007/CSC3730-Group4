"""
Command-line interface for generating movie recommendations and comparing models.

Examples:
    python recommend.py --user 1 --model all --profile
    python recommend.py --user 42 --model hybrid --alpha 0.3 --n 10
    python recommend.py --evaluate
"""

import argparse
from pathlib import Path
import sys

from src.content_based import ContentBasedRecommender
from src.data import DEFAULT_DATA_DIR, load_movielens
from src.evaluation import (
    compare_models,
    compare_user_recommendations,
    train_test_split_by_time,
    user_genre_profile,
)
from src.hybrid import HybridRecommender
from src.popularity import PopularityRecommender
from src.random_baseline import RandomRecommender


def build_parser():
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="MovieLens 100K movie recommender and evaluation CLI."
    )
    parser.add_argument(
        "--user",
        type=int,
        default=1,
        help="User ID to generate recommendations for (default: 1).",
    )
    parser.add_argument(
        "--model",
        choices=["content", "popularity", "hybrid", "random", "all"],
        default="all",
        help="Recommender model to run, or 'all' for side-by-side comparison (default: all).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid weight on content-based score from 0 to 1 (default: 0.5).",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=50,
        help="Minimum rating count required by popularity/hybrid models (default: 50).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Minimum rating considered positive, from 1 to 5 (default: 4).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Display the target user's rating summary, top liked genres, and top movies.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run an 80/20 time-split offline comparison across all four models.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the extracted MovieLens 100K directory.",
    )
    return parser


def _build_models(movies, alpha, min_ratings):
    return {
        "Random (floor)": RandomRecommender(movies, seed=0),
        "Content-based": ContentBasedRecommender(movies),
        "Popularity": PopularityRecommender(movies, min_ratings=min_ratings),
        f"Hybrid (alpha={alpha:g})": HybridRecommender(
            movies, alpha=alpha, min_ratings=min_ratings
        ),
    }


def _print_user_profile(ratings, movies, user_id, threshold):
    profile = user_genre_profile(ratings, movies, user_id=user_id, threshold=threshold)
    top_genres = profile["genre_counts"].head(5)
    genres_str = (
        ", ".join(f"{genre} ({count})" for genre, count in top_genres.items())
        if not top_genres.empty
        else "None"
    )
    print(f"=== User {user_id} Profile ===")
    print(
        f"Ratings: {profile['total_ratings']} total | "
        f"{profile['positive_ratings']} positive (>= {threshold}) | "
        f"mean rating: {profile['mean_rating']:.2f}"
    )
    print(f"Top liked genres: {genres_str}")

    sort_cols = ["rating", "timestamp", "movie_id"] if "timestamp" in ratings.columns else ["rating", "movie_id"]
    top_rated = (
        ratings[ratings["user_id"] == user_id]
        .sort_values(sort_cols, ascending=[False] * (len(sort_cols) - 1) + [True])
        .head(5)
        .merge(movies[["movie_id", "title"]], on="movie_id", how="left")
    )
    print("Recently top-rated movies:")
    for _, row in top_rated.iterrows():
        print(f"  - [{int(row['rating'])}*] {row['title']}")
    print()


def main(argv=None):
    """Run the movie recommendation CLI and return an exit status code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        ratings, movies, _ = load_movielens(args.data_dir)
        models = _build_models(movies, alpha=args.alpha, min_ratings=args.min_ratings)

        if args.evaluate:
            train, test = train_test_split_by_time(ratings, test_fraction=0.2)
            summary = compare_models(
                models, train, test, k=args.n, threshold=args.threshold
            )
            print(f"=== Offline Evaluation (80/20 time split, k={args.n}, threshold={args.threshold}) ===")
            print(summary.round(4).to_string())
            return 0

        if args.profile:
            _print_user_profile(ratings, movies, user_id=args.user, threshold=args.threshold)

        if args.model == "all":
            comparison = compare_user_recommendations(
                models,
                ratings,
                movies,
                user_id=args.user,
                n=args.n,
                threshold=args.threshold,
            )
            print(f"=== Top-{args.n} Recommendations for User {args.user} ===")
            print(comparison.to_string())
            return 0

        key_map = {
            "random": "Random (floor)",
            "content": "Content-based",
            "popularity": "Popularity",
            "hybrid": f"Hybrid (alpha={args.alpha:g})",
        }
        chosen_name = key_map[args.model]
        recs = models[chosen_name].recommend(
            ratings, user_id=args.user, n=args.n, threshold=args.threshold
        )
        print(f"=== {chosen_name} Top-{args.n} for User {args.user} ===")
        print(recs.to_string(index=False))
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
