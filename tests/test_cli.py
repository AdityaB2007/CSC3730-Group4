from recommend import main
from src.data import GENRE_COLS


def _write_synthetic_movielens(folder):
    rows = []
    # Users 1..4 with >= 5 timestamped ratings so 80/20 time split leaves train & test.
    ratings_data = [
        (1, 1, 5, 10), (1, 2, 5, 20), (1, 3, 2, 30), (1, 4, 4, 40), (1, 5, 5, 50),
        (2, 1, 4, 10), (2, 2, 3, 20), (2, 6, 5, 30), (2, 7, 5, 40), (2, 8, 4, 50),
        (3, 3, 5, 10), (3, 4, 5, 20), (3, 6, 4, 30), (3, 7, 4, 40), (3, 9, 5, 50),
        (4, 2, 4, 10), (4, 5, 5, 20), (4, 8, 5, 30), (4, 9, 4, 40), (4, 10, 5, 50),
    ]
    (folder / "u.data").write_text(
        "\n".join(f"{u}\t{m}\t{r}\t{t}" for u, m, r, t in ratings_data) + "\n",
        encoding="utf-8",
    )

    item_lines = []
    for mid in range(1, 13):
        flags = [0] * len(GENRE_COLS)
        flags[1 if mid % 2 else 5] = 1  # alternating Action / Comedy
        genre_str = "|".join(str(x) for x in flags)
        item_lines.append(
            f"{mid}|Movie {mid} (1995)|01-Jan-1995||http://example.com/{mid}|{genre_str}"
        )
    (folder / "u.item").write_text("\n".join(item_lines) + "\n", encoding="latin-1")

    user_lines = [f"{uid}|25|M|student|1000{uid}" for uid in range(1, 5)]
    (folder / "u.user").write_text("\n".join(user_lines) + "\n", encoding="utf-8")


def test_cli_all_models_with_profile(tmp_path, capsys):
    _write_synthetic_movielens(tmp_path)
    code = main([
        "--data-dir", str(tmp_path),
        "--user", "1",
        "--model", "all",
        "--min-ratings", "1",
        "--n", "2",
        "--profile",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "User 1 Profile" in out
    assert "Top liked genres:" in out
    assert "Top-2 Recommendations for User 1" in out
    assert "Hybrid (alpha=0.5)" in out


def test_cli_single_hybrid_model(tmp_path, capsys):
    _write_synthetic_movielens(tmp_path)
    code = main([
        "--data-dir", str(tmp_path),
        "--user", "1",
        "--model", "hybrid",
        "--alpha", "0.3",
        "--min-ratings", "1",
        "--n", "3",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Hybrid (alpha=0.3) Top-3 for User 1" in out


def test_cli_evaluate_flag(tmp_path, capsys):
    _write_synthetic_movielens(tmp_path)
    code = main([
        "--data-dir", str(tmp_path),
        "--evaluate",
        "--min-ratings", "1",
        "--n", "2",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Offline Evaluation" in out
    assert "precision@2" in out
    assert "ndcg@2" in out
    assert "personalization" in out


def test_cli_invalid_user_returns_nonzero(tmp_path, capsys):
    _write_synthetic_movielens(tmp_path)
    code = main([
        "--data-dir", str(tmp_path),
        "--user", "999",
        "--model", "content",
    ])
    err = capsys.readouterr().err
    assert code == 1
    assert "Error:" in err
