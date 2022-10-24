import numpy as np

def split_df(data, ratios, shuffle=False):
    assert np.sum(ratios) == 1, "sum of the raios should be 1"
    split_index = np.cumsum(ratios).tolist()[:-1]

    if shuffle:
        data = data.sample(frac=1)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    return splits