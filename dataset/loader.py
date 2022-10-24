from pathlib import Path

import pandas as pd
import scipy.sparse as ssp


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
    def get_df(cls, size="ml-100k"):
        data_infos = MOVIELENSE_DATA_INFOS[size]
        rating_infos = data_infos["rating"]
        rating_df = pd.read_csv(
            rating_infos["path"],
            delimiter=rating_infos["delimiter"],
            header=rating_infos["header"],
            names=rating_infos["header_names"],
            dtype=rating_infos["dtype"],
        )
        return rating_df


    @classmethod
    def get_df_cached(cls, size="ml-100k"):
        data_infos = MOVIELENSE_DATA_INFOS[size]
        rating_infos = data_infos["rating"]
        cached_path = Path(rating_infos["cached_path"])
        if cached_path.exists():
            return pd.read_feather(cached_path)
        else:
            rating_df = pd.read_csv(
                rating_infos["path"],
                delimiter=rating_infos["delimiter"],
                header=rating_infos["header"],
                names=rating_infos["header_names"],
                dtype=rating_infos["dtype"],
            )
            rating_df.to_feather(cached_path)
        return rating_df

    @classmethod
    def get_matrix(cls, size="ml-100k", rating_df=None, sparse=True):
        if rating_df is None:
            rating_df = cls.get_df(size)
        num_users = rating_df.user_id.max()
        num_items = rating_df.item_id.max()
        ratings, users, items = [], [], []
        for row in rating_df.itertuples():
            users.append(row.user_id)
            items.append(row.item_id)
            ratings.append(row.rating)
        sparse_matrix = ssp.csr_matrix(
            (ratings, (users, items)),
            shape=(num_users + 1, num_items + 1)
        )
        if sparse:
            return sparse_matrix
        return sparse_matrix.todense()

