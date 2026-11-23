# Movie Recommendation System - Fall 2026 ML Project

## Abstract

This project investigates the effectiveness of popularity-based, content-based, and hybrid approaches to movie recommendation on the GroupLens MovieLens 100K dataset (`100,000` ratings from `943` users across `1,682` movies; `93.70%` interaction sparsity). The popularity-based approach ranks movies with at least `min_ratings` ratings by their proportion of positive ratings (`rating >= 4`), while the personalized content-based model computes cosine similarity across 19 binary movie genre indicators with rating-count tie-breaking. The hybrid approach blends min-max normalized content and popularity scores via `score = alpha * content_score + (1 - alpha) * popularity_score`.

All models were evaluated against a reproducible random baseline floor using an 80/20 per-user time-based hold-out split (`80,000` training ratings, `20,000` test ratings, `906` evaluated users), across an 11-point `alpha` sweep (`0.0` to `1.0`), hold-out fraction robustness checks (`0.1`, `0.2`, `0.3`), `min_ratings` sensitivity thresholds, and user-activity segments. Empirically, the hybrid recommender outperformed every individual approach across all accuracy and ranking metrics: at `alpha = 0.3` it peaked at `precision@10 = 0.0442` and `mrr@10 = 0.1249` (vs. Popularity `0.0350` / `0.0542`, Content-based `0.0243` / `0.0837`, and Random `0.0074` / `0.0220`), while `alpha = 0.5` reached `precision@10 = 0.0425`, `hit_rate@10 = 0.2903`, `ndcg@10 = 0.0544`, and `personalization = 0.7394`. Contrary to our initial hypothesis that pure content-based recommendations would outperform popularity overall, 19 binary genre flags proved too coarse to beat popularity on `precision@10` alone—yet content similarity improved `mrr@10`, catalog coverage (`0.2747` vs. `0.0309`), and personalization (`0.8833` vs. `0.2891`), making the hybrid blend strictly superior to either standalone model.

### Contributors
<ul>
  <li>Seth Auzenne</li>
  <li>Aditya Baisakh</li>
  <li>Nicolas Chrest</li>
  <li>Stella Levy</li>
  <li>Keith Mayle</li>
</ul>

## Getting Started

Requires Python 3.10 or newer. Run these from the project root.

```bash
# 1. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the MovieLens 100K dataset into ./ml-100k
python download_data.py

# 4. Open the notebooks
jupyter notebook
```

If you'd rather download the data by hand, get `ml-100k.zip` from
[GroupLens](https://grouplens.org/datasets/movielens/100k/) and unzip it so that
`ml-100k/u.data` sits in the project root. The `ml-100k/` folder is gitignored.

## Methodology

- **Random baseline (`RandomRecommender`)**: Uniformly samples unrated movies using a per-user deterministic seed as the accuracy floor.
- **Popularity (`PopularityRecommender`)**: Filters movies with at least `min_ratings` ratings (`50` by default, leaving `603` of `1,682` movies covering `83.72%` of ratings) and ranks by `positive_ratio = count(rating >= 4) / count(ratings)`, breaking ties by rating count and `movie_id`.
- **Content-based (`ContentBasedRecommender`)**: Represents each movie by its 19 binary `GENRE_COLS` vector, computes movie-to-movie cosine similarity, scores unrated candidates by mean similarity to the user's positively rated (`>= 4`) movies, and breaks similarity ties by rating count and `movie_id`.
- **Hybrid (`HybridRecommender`)**: Min-max rescales candidate content and popularity scores to `[0, 1]` per user and ranks by `alpha * content_score + (1 - alpha) * popularity_score`.
- **Evaluation (`src/evaluation.py`)**: Holds out each user's most recent 20% of ratings (`train_test_split_by_time`) so no future ratings leak into training, and evaluates `precision@10`, `recall@10`, `hit_rate@10`, `ndcg@10`, `mrr@10`, catalog `coverage`, and `personalization` (`1 - mean pairwise cosine similarity` across user lists).

## Key Results

Results on the 80/20 time-based split (`906` evaluated users, `k=10`, `threshold=4`, `min_ratings=50`):

| Model | `precision@10` | `recall@10` | `hit_rate@10` | `ndcg@10` | `mrr@10` | `coverage` | `personalization` |
|-------|---------------:|------------:|--------------:|----------:|---------:|-----------:|------------------:|
| Random (floor) | `0.0074` | `0.0061` | `0.0651` | `0.0085` | `0.0220` | `0.9941` | `0.9940` |
| Content-based | `0.0243` | `0.0242` | `0.1865` | `0.0332` | `0.0837` | `0.2747` | `0.8833` |
| Popularity (`min_ratings=50`) | `0.0350` | `0.0296` | `0.2252` | `0.0330` | `0.0542` | `0.0309` | `0.2891` |
| Hybrid (`alpha=0.5`) | `0.0425` | `0.0428` | **`0.2903`** | `0.0544` | `0.1220` | `0.1094` | `0.7394` |
| Hybrid (`alpha=0.3`, tuned) | **`0.0442`** | `0.0406` | `0.2770` | **`0.0552`** | **`0.1249`** | `0.0654` | `0.6264` |

- **Alpha trade-off**: Shifting `alpha` from `0.0` to `1.0` monotonically increases catalog coverage (`0.0309 -> 0.2747`) and personalization (`0.2891 -> 0.8833`), while blending in `[0.3, 0.5]` beats both endpoints on every accuracy and ranking metric (`alpha=0.4` peaks on `recall@10 = 0.0437` and `ndcg@10 = 0.0553`).
- **Robustness**: Model rankings on `precision@10` (`Hybrid > Popularity > Content-based > Random`) remained unchanged across `test_fraction` values of `0.1`, `0.2`, and `0.3`.
- **Activity segments**: On light users (`16–35` training ratings), Content-based (`precision@10 = 0.0134`) outperformed Popularity (`0.0098`) and matched Hybrid (`0.0130`), whereas Hybrid widened its lead among medium (`0.0424`) and heavy (`0.0775`) users.
- **Data quality (`eda.ipynb`)**: Zero duplicate rating records, zero invalid ratings outside `[1, 5]`, and zero orphan IDs; `55.38%` of ratings are positive (`>= 4`), with `18` duplicate movie titles sharing distinct `movie_id`s.

## Project Structure

| Path | What it is |
|------|------------|
| `main.ipynb` | Executed modeling, alpha sweep, sensitivity, segment, and qualitative comparisons |
| `eda.ipynb` | Executed exploratory data analysis and modeling-readiness checks on MovieLens 100K |
| `recommend.py` | CLI tool to inspect user profiles, generate recommendations, and run offline evaluation |
| `src/data.py` | Downloads and loads the dataset (`load_movielens`, `download_movielens`) |
| `src/content_based.py` | `ContentBasedRecommender`: genre cosine similarity with rating-count tie-breaking |
| `src/popularity.py` | `PopularityRecommender`: most-rated, most-liked baseline |
| `src/hybrid.py` | `HybridRecommender`: weighted blend of min-max normalized scores |
| `src/random_baseline.py` | `RandomRecommender`: random unrated movies, the floor every model should beat |
| `src/evaluation.py` | Time-based train/test split, ranking/diversity metrics, segment and user helpers |
| `tests/` | Unit tests (run with `pytest`) |
| `download_data.py` | Script that fetches the dataset |

## Usage

### Python API

```python
from src.data import load_movielens
from src.content_based import ContentBasedRecommender
from src.popularity import PopularityRecommender
from src.hybrid import HybridRecommender

ratings, movies, users = load_movielens()

ContentBasedRecommender(movies).recommend(ratings, user_id=1, n=10, threshold=4)
PopularityRecommender(movies, min_ratings=50).recommend(ratings, user_id=1, n=10)
HybridRecommender(movies, alpha=0.3, min_ratings=50).recommend(ratings, user_id=1, n=10)
```

To compare them on an 80/20 time-based split:

```python
from src.evaluation import train_test_split_by_time, compare_models

train, test = train_test_split_by_time(ratings, test_fraction=0.2)
compare_models(
    {"Content-based": ContentBasedRecommender(movies),
     "Popularity": PopularityRecommender(movies),
     "Hybrid": HybridRecommender(movies, alpha=0.3)},
    train, test, k=10,
)
```

### Command-Line Interface

```bash
# Side-by-side recommendations with user profile summary
python recommend.py --user 1 --model all --profile

# Tuned hybrid recommendations for a specific user
python recommend.py --user 316 --model hybrid --alpha 0.3 --n 10

# Full 80/20 offline comparison table
python recommend.py --evaluate
```

## Reproducing the Results

Both [`eda.ipynb`](eda.ipynb) and [`main.ipynb`](main.ipynb) are committed with saved outputs and charts so results are readable directly on GitHub. To re-run everything from scratch:

1. Download the dataset with `python download_data.py`.
2. Open `eda.ipynb` and `main.ipynb` in Jupyter and run **Kernel → Restart & Run All**.

## Running Tests

```bash
pytest -v
```

Tests also run automatically on every push and pull request through GitHub Actions.
