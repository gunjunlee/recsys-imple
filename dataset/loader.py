from pathlib import Path

import pandas as pd


MOVIELENSE_RATING_HEADER = ["user_id", "item_id", "rating", "timestamp"]
MOVIELENSE_RATING_DTYPE = {"user_id": int, "item_id": int, "rating": float, "timestamp": int}

MOVIELENSE_DATA_INFOS = {
    "MovieLense100K": {
        "rating": {
            "path": "data/ml-100k/u.data",
            "cached_path": "data/ml-100k/rating.feather",
            "delimiter": "\t",
            "header": None,
            "header_names": MOVIELENSE_RATING_HEADER,
            "dtype": MOVIELENSE_RATING_DTYPE,
        },
    },
    "MovieLense1M": {
        "rating": {
            "path": "data/ml-1m/ratings.dat",
            "cached_path": "data/ml-1m/rating.feather",
            "delimiter": "::",
            "header": None,
            "header_names": MOVIELENSE_RATING_HEADER,
            "dtype": MOVIELENSE_RATING_DTYPE,
        },
    },
    "MovieLense10M": {
        "rating": {
            "path": "data/ml-10M100K/ratings.dat",
            "cached_path": "data/ml-10M100K/rating.feather",
            "delimiter": "::",
            "header": None,
            "header_names": MOVIELENSE_RATING_HEADER,
            "dtype": MOVIELENSE_RATING_DTYPE,
        },
    },
    "MovieLense20M": {
        "rating": {
            "path": "data/ml-20m/ratings.csv",
            "cached_path": "data/ml-20m/rating.feather",
            "delimiter": ",",
            "header": 0,
            "header_names": MOVIELENSE_RATING_HEADER,
            "dtype": MOVIELENSE_RATING_DTYPE,
        },
    },
    "MovieLense25M": {
        "rating": {
            "path": "data/ml-25m/ratings.csv",
            "cached_path": "data/ml-25m/rating.feather",
            "delimiter": ",",
            "header": 0,
            "header_names": MOVIELENSE_RATING_HEADER,
            "dtype": MOVIELENSE_RATING_DTYPE,
        },
    },
}

class MovieLense:
    @classmethod
    def get_df(cls, size="ml-100k", cache=True):
        data_infos = MOVIELENSE_DATA_INFOS[size]
        rating_infos = data_infos["rating"]
        if cache:
            cached_path = Path(rating_infos["cached_path"])
            if cached_path.exists():
                return pd.read_feather(cached_path)
        rating_df = pd.read_csv(
            rating_infos["path"],
            delimiter=rating_infos["delimiter"],
            header=rating_infos["header"],
            names=rating_infos["header_names"],
            dtype=rating_infos["dtype"],
        )
        if cache:
            rating_df.to_feather(cached_path)
        return rating_df
