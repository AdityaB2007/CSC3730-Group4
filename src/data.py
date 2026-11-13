"""
The purpose of this file is to handle consistent data loading.
"""

from pathlib import Path
import io
import urllib.request
import zipfile

import pandas as pd

# Repository root, so the default data path works no matter which folder
# a notebook or script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "ml-100k"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
REQUIRED_FILES = ("u.data", "u.item", "u.user")

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

def download_movielens(data_dir=DEFAULT_DATA_DIR, url=MOVIELENS_URL):
    """
    Download and extract MovieLens 100K into data_dir.
    Does nothing if the dataset is already there.
    return: Path to the dataset folder
    """
    data_dir = Path(data_dir)
    if all((data_dir / name).exists() for name in REQUIRED_FILES):
        print(f"MovieLens 100K already present in {data_dir}")
        return data_dir

    print(f"Downloading MovieLens 100K from {url} ...")
    context = None
    try:
        import certifi
        import ssl
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            import ssl
            context = ssl.create_default_context()
        except Exception:
            context = None

    try:
        response = urllib.request.urlopen(url, context=context) if context else urllib.request.urlopen(url)
    except Exception:
        import ssl
        unverified = ssl._create_unverified_context()
        response = urllib.request.urlopen(url, context=unverified)

    with response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    # The zip contains a top-level "ml-100k/" folder; extract next to it.
    archive.extractall(data_dir.parent)
    extracted = data_dir.parent / "ml-100k"
    if extracted != data_dir:
        extracted.rename(data_dir)

    print(f"Saved dataset to {data_dir}")
    return data_dir


def load_movielens(data_dir=DEFAULT_DATA_DIR):
    """
    Load MovieLens 100K ratings, movies, and users.
    return: ratings, movies, users
    """
    data_dir = Path(data_dir)
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"MovieLens 100K not found in {data_dir} (missing {', '.join(missing)}). "
            "Run `python download_data.py` from the project root first."
        )
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
            "release_date",
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