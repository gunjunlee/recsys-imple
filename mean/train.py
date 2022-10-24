import logging

import hydra
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from utils.utils import Timer
from utils.utils import fix_rng
from dataset.utils import split_df
from dataset.loader import MovieLense


log = logging.getLogger(__name__)


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = torch.sqrt(torch.mean((output - target)**2))
        return loss


@hydra.main(config_path="conf", config_name="train")
def main(config):
    log.info(OmegaConf.to_yaml(config))
    if config.main.reproducibility.fix_rng:
        fix_rng(config.main.reproducibility.seed)

    with Timer("read df"):
        df = MovieLense.get_df_cached(config.main.dataset)
    train_df, valid_df = split_df(df, [0.90, 0.10], shuffle=True)
    user_mean = train_df.groupby("user_id")["rating"].mean().to_dict()
    item_mean = train_df.groupby("item_id")["rating"].mean().to_dict()
    global_mean = train_df["rating"].mean()

    target = []
    user_output = []
    item_output = []
    for row in valid_df.itertuples():
        target.append(row.rating)
        user_output.append(user_mean.get(row.user_id, global_mean))
        item_output.append(item_mean.get(row.item_id, global_mean))

    target = torch.tensor(target)
    user_output = torch.tensor(user_output)
    item_output = torch.tensor(item_output)

    user_rmse = RMSE()(user_output, target).item()
    item_rmse = RMSE()(item_output, target).item()
    log.info(f"user mean RMSE: {user_rmse:.4f}")
    log.info(f"item mean RMSE: {item_rmse:.4f}")


if __name__ == "__main__":
    main()
