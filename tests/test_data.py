import pytest

from src.data import GENRE_COLS, load_movielens


def write_tiny_dataset(folder):
    genres = "|".join(["0"] * len(GENRE_COLS))
    (folder / "u.data").write_text("1\t10\t5\t881250949\n2\t10\t3\t881250950\n")
    (folder / "u.item").write_text(
        f"10|Toy Story (1995)|01-Jan-1995||http://example.com|{genres}\n",
        encoding="latin-1",
    )
    (folder / "u.user").write_text(
        "1|24|M|technician|85711\n2|53|F|other|94043\n"
    )


def test_load_movielens(tmp_path):
    write_tiny_dataset(tmp_path)
    ratings, movies, users = load_movielens(tmp_path)

    assert list(ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]
    assert len(ratings) == 2
    assert movies.loc[0, "title"] == "Toy Story (1995)"
    assert "release_date" in movies.columns
    assert set(GENRE_COLS) <= set(movies.columns)
    assert len(users) == 2


def test_missing_dataset_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_movielens(tmp_path / "nowhere")
