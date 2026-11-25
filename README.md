# Movie Recommendation System - Fall 2026 ML Project

## Abstract

This project compares popularity-based, content-based, and hybrid approaches to movie recommendation on the GroupLens MovieLens 100K dataset, which contains 100,000 ratings from 943 users across 1,682 movies (93.70% sparse). The popularity-based approach ranks movies with at least `min_ratings` ratings according to their proportion of positive ratings (4 stars or higher), while the personalized approach recommends unrated movies using cosine similarity across 19 binary genre indicators with rating-count tie-breaking. The hybrid approach combines min-max scaled content and popularity scores with weight `alpha` to balance broad popularity and individual genre preferences.

All models were evaluated against a random baseline on an 80/20 time-based hold-out split (80,000 training ratings, 20,000 test ratings, 906 evaluated users), across an `alpha` sweep from 0.0 to 1.0, across multiple hold-out sizes (0.1, 0.2, 0.3), and across user activity levels. The hybrid recommender outperformed both standalone approaches across every accuracy and ranking metric: `alpha=0.3` reached the highest `precision@10` (0.0442) and `mrr@10` (0.1249), while `alpha=0.5` reached `precision@10` of 0.0425, `hit_rate@10` of 0.2903, `ndcg@10` of 0.0544, and `personalization` of 0.7394, compared with Popularity (`precision@10` 0.0350) and Content-based (`precision@10` 0.0243). Pure genre-based recommendations scored lower than Popularity on `precision@10` because 19 binary genre tags are fairly coarse, though Content-based achieved higher `mrr@10` (0.0837 vs. 0.0542), catalog `coverage` (0.2747 vs. 0.0309), and `personalization` (0.8833 vs. 0.2891).

### Team Members (Group 4)
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

## Project Structure

| Path | What it is |
|------|------------|
| `main.ipynb` | Runs all recommenders, compares them, and evaluates sensitivity and user segments |
| `eda.ipynb` | Exploratory data analysis of MovieLens 100K |
| `src/data.py` | Downloads and loads the dataset (`load_movielens`, `download_movielens`) |
| `src/content_based.py` | `ContentBasedRecommender`: genre cosine similarity |
| `src/popularity.py` | `PopularityRecommender`: most-rated, most-liked baseline |
| `src/hybrid.py` | `HybridRecommender`: weighted blend of the two scores |
| `src/random_baseline.py` | `RandomRecommender`: random unrated movies, the floor every model should beat |
| `src/evaluation.py` | Time-based split, precision/recall/hit-rate/NDCG/MRR/coverage/personalization metrics, and activity breakdowns |
| `tests/` | Unit tests (run with `pytest`) |
| `download_data.py` | Script that fetches the dataset |

## Results

Results on the 80/20 time-based split (906 evaluated users, `k=10`, `threshold=4`, `min_ratings=50`):

| Model | precision@10 | recall@10 | hit_rate@10 | ndcg@10 | mrr@10 | coverage | personalization |
|-------|-------------:|----------:|------------:|--------:|-------:|---------:|----------------:|
| Random (floor) | 0.0074 | 0.0061 | 0.0651 | 0.0085 | 0.0220 | 0.9941 | 0.9940 |
| Content-based | 0.0243 | 0.0242 | 0.1865 | 0.0332 | 0.0837 | 0.2747 | 0.8833 |
| Popularity (`min_ratings=50`) | 0.0350 | 0.0296 | 0.2252 | 0.0330 | 0.0542 | 0.0309 | 0.2891 |
| Hybrid (`alpha=0.5`) | 0.0425 | 0.0428 | 0.2903 | 0.0544 | 0.1220 | 0.1094 | 0.7394 |
| Hybrid (`alpha=0.3`, best precision) | 0.0442 | 0.0406 | 0.2770 | 0.0552 | 0.1249 | 0.0654 | 0.6264 |

Moving `alpha` from 0.0 to 1.0 raises catalog `coverage` (0.0309 to 0.2747) and `personalization` (0.2891 to 0.8833), while `precision@10` peaks around `alpha=0.3` and `ndcg@10` peaks at `alpha=0.4` (0.0553). Model rankings by `precision@10` stayed consistent across `test_fraction` values of 0.1, 0.2, and 0.3, and Hybrid stayed ahead of Popularity across `min_ratings` values from 10 to 200.

Breaking users into tertiles by training history length shows that Content-based (`precision@10` 0.0134) outperformed Popularity (0.0098) for light users (16-35 training ratings), whereas Hybrid (`alpha=0.3`) led on medium (0.0424) and heavy (0.0775) users. In `eda.ipynb`, all 100,000 ratings had valid values (1-5) with no duplicate `(user_id, movie_id)` pairs, 55.38% positive ratings (`>= 4`), and 603 of 1,682 movies qualifying at `min_ratings=50`.

## Usage

All three recommenders share the same interface:

```python
from src.data import load_movielens
from src.content_based import ContentBasedRecommender
from src.popularity import PopularityRecommender
from src.hybrid import HybridRecommender

ratings, movies, users = load_movielens()

ContentBasedRecommender(movies).recommend(ratings, user_id=1, n=10, threshold=4)
PopularityRecommender(movies, min_ratings=50).recommend(ratings, user_id=1, n=10)
HybridRecommender(movies, alpha=0.3).recommend(ratings, user_id=1, n=10)
```

To compare them, hold out each user's most recent ratings and score the top-k lists:

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

## Reproducing the Results

Both `eda.ipynb` and `main.ipynb` are saved with their outputs so the charts and tables render on GitHub. To re-run them locally, run `python download_data.py`, open either notebook in Jupyter, and run **Restart & Run All** from the Kernel menu.

## Running Tests

```bash
pytest
```

Tests also run automatically on every push and pull request through GitHub Actions.
