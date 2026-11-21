# Movie Recommendation System - Fall 2026 ML Project

## Abstract

This project investigates the effectiveness of popularity-based, personalized, and hybrid approaches to movie recommendation. The popularity-based approach ranks movies according to their rating frequency and proportion of positive ratings, while the personalized approach recommends movies based on genre similarities between movies a user has previously rated positively and movies they have not yet rated. The hybrid approach combines the popularity and personalized recommendation scores using different weighting factors to investigate how the balance between overall popularity and individual preferences affects recommendation performance.

The project uses the GroupLens MovieLens 100K dataset, consisting of 100,000 ratings from 943 users across 1,682 movies. For personalized recommendations, cosine similarity is used to compare movies based on their genre classifications and identify unrated movies that are similar to a user's positively rated movies.

The three approaches will be evaluated using a consistent training and testing methodology. For the hybrid approach, multiple weighting combinations will be tested to determine how different contributions from popularity and personalized scores affect recommendation performance. The results will be compared to assess whether combining both approaches provides an improvement over either method individually.

We hypothesize that personalized recommendations will better reflect individual user preferences than popularity-based recommendations, while the hybrid approach may provide a balance between individual relevance and broader user preferences.

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

## Project Structure

| Path | What it is |
|------|------------|
| `main.ipynb` | Runs all three recommenders, compares them, and tunes the hybrid weight |
| `eda.ipynb` | Exploratory data analysis of MovieLens 100K |
| `src/data.py` | Downloads and loads the dataset (`load_movielens`, `download_movielens`) |
| `src/content_based.py` | `ContentBasedRecommender`: genre cosine similarity |
| `src/popularity.py` | `PopularityRecommender`: most-rated, most-liked baseline |
| `src/hybrid.py` | `HybridRecommender`: weighted blend of the two scores |
| `src/evaluation.py` | Time-based train/test split and precision/recall/hit-rate/coverage metrics |
| `tests/` | Unit tests (run with `pytest`) |
| `download_data.py` | Script that fetches the dataset |

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
HybridRecommender(movies, alpha=0.5).recommend(ratings, user_id=1, n=10)
```

To compare them, hold out each user's most recent ratings and score the top-k lists:

```python
from src.evaluation import train_test_split_by_time, compare_models

train, test = train_test_split_by_time(ratings, test_fraction=0.2)
compare_models(
    {"Content-based": ContentBasedRecommender(movies),
     "Popularity": PopularityRecommender(movies),
     "Hybrid": HybridRecommender(movies, alpha=0.5)},
    train, test, k=10,
)
```

## Running Tests

```bash
pytest
```

Tests also run automatically on every push and pull request through GitHub Actions.
