"""
Download the MovieLens 100K dataset into ./ml-100k.

Usage (from the project root):
    python download_data.py
"""

from src.data import download_movielens

if __name__ == "__main__":
    download_movielens()
