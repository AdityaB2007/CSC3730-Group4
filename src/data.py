"""
The purpose of this file is to handle consistent data loading.
"""

from pathlib import Path
import pandas as pd

GENRE_COLS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]

def load_movielens(data_dir="./ml-100k"):
    """
    Load MovieLens 100K ratings, movies, and users.
    return: ratings, movies, users
    """
    data_dir = Path(data_dir)
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        names=[
            "movie_id",
            "title",
            "release_data",
            "video_release_date",
            "imdb_url",
            *GENRE_COLS
        ]
    )

    users = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=[
            "user_id",
            "age",
            "gender",
            "occupation",
            "zip_code"
        ]
    )

    return ratings, movies, users