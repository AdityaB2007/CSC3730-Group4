# Movie Recommendation System - Fall 2026 ML Project

## Abstract:
This project will compare two approaches to movie recommendations: popularity-based recommendations and personalized recommendations. The popularity-based approach will recommend movies that have been rated by a large portion of users and have a high proportion of positive ratings. The personalized approach will recommend movies based on genre similarities between movies a user has previously rated (positively) and movies that user has not yet rated. Comparing these approaches will help us better understand how to weigh popularity and personalized preferences when developing an optimized recommendation algorithm.
<br><br>
The project will use the GroupLens MovieLens 100K dataset, which contains 100,000 movie ratings from 943 users across 1,682 movies.
<br><br>
The model will use cosine similarity to compare movies based on the similarities in their subsets of genre classifications. It will use these similarity scores to recommend unrated movies that are similar to movies the user has rated positively. The results will be compared against the popularity-based approach.
<br><br>
We expect personalized recommendations to be more relevant to individual users than recommendations based solely on popularity. The results may help us determine how to balance popularity and personalized preferences in a recommendation algorithm.

### Contributors:
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
| `main.ipynb` | Runs both recommenders and compares them |
| `eda.ipynb` | Exploratory data analysis of MovieLens 100K |
| `src/data.py` | Downloads and loads the dataset (`load_movielens`, `download_movielens`) |
| `src/content_based.py` | `ContentBasedRecommender`: genre cosine similarity |
| `src/popularity.py` | `PopularityRecommender`: most-rated, most-liked baseline |
| `src/evaluation.py` | Time-based train/test split and precision/recall/hit-rate/coverage metrics |
| `tests/` | Unit tests (run with `pytest`) |
| `download_data.py` | Script that fetches the dataset |

## Usage

Both recommenders share the same interface:

```python
from src.data import load_movielens
from src.content_based import ContentBasedRecommender
from src.popularity import PopularityRecommender

ratings, movies, users = load_movielens()

ContentBasedRecommender(movies).recommend(ratings, user_id=1, n=10, threshold=4)
PopularityRecommender(movies, min_ratings=50).recommend(ratings, user_id=1, n=10)
```

To compare them, hold out each user's most recent ratings and score the top-k lists:

```python
from src.evaluation import train_test_split_by_time, compare_models

train, test = train_test_split_by_time(ratings, test_fraction=0.2)
compare_models(
    {"Content-based": ContentBasedRecommender(movies),
     "Popularity": PopularityRecommender(movies)},
    train, test, k=10,
)
```

## Running Tests

```bash
pytest
```

Tests also run automatically on every push and pull request through GitHub Actions.
