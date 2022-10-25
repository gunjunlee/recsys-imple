import logging

import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from utils.utils import fix_rng
from dataset.utils import split_df
from dataset.loader import MovieLense


log = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, user_id_converter, item_id_converter):
        super().__init__()
        self.df = df
        self.user_id_converter = user_id_converter
        self.item_id_converter = item_id_converter
        self.user_ids = []
        self.item_ids = []
        self.ratings = []
        for row in self.df.itertuples():
            user_id = self.user_id_converter.get(row.user_id, 0)
            item_id = self.item_id_converter.get(row.item_id, 0)
            self.user_ids.append(user_id)
            self.item_ids.append(item_id)
            self.ratings.append(row.rating)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        rating = self.ratings[idx]
        return {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
        }


class Model(nn.Module):
    def __init__(self, mu, num_users, num_items, num_factors, dropout=0.2):
        super().__init__()
        self.mu = mu
        self.num_factors = num_factors
        self.user_bias = nn.Embedding(num_users + 1, 1)
        self.item_bias = nn.Embedding(num_items + 1, 1)
        self.user_embedding = nn.Embedding(num_users + 1, num_factors)
        self.item_embedding = nn.Embedding(num_items + 1, num_factors)
        self.dropout = dropout

    def forward(self, user_ids, item_ids):
        b = user_ids.size(0)
        if self.training:
            user_ids = torch.empty_like(user_ids).bernoulli_(1 - self.dropout) * user_ids
            item_ids = torch.empty_like(item_ids).bernoulli_(1 - self.dropout) * item_ids

        user_bias = self.user_bias(user_ids).view(b)
        item_bias = self.item_bias(item_ids).view(b)
        user_embeddings = self.user_embedding(user_ids).view(b, 1, -1)
        item_embeddings = self.item_embedding(item_ids).view(b, -1, 1)

        score = self.mu + user_bias + item_bias + torch.bmm(user_embeddings, item_embeddings).view(b)

        return score, user_bias, item_bias, user_embeddings, item_embeddings


class RMSE(nn.Module):
    def __init__(self, min=1., max=5.):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, output, target):
        output = torch.clamp(output, min=self.min, max=self.max)
        loss = torch.sqrt(torch.mean((output - target)**2))
        return loss


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = torch.mean((output - target)**2)
        return loss


def train(model, optimizer, criteria, train_dataloader, bias_weight_decay=1.0, embd_weight_decay=1.0):
    model = model.train()
    losses = []
    with tqdm(train_dataloader, dynamic_ncols=True) as pbar:
        for data in pbar:
            user_id = data["user_id"].cuda()
            item_id = data["item_id"].cuda()
            rating = data["rating"].cuda()

            output, user_bias, item_bias, user_embeddings, item_embeddings = model(user_id, item_id)

            loss = criteria(output, rating)
            optimizer.zero_grad()

            bias_reg = (user_bias ** 2).sum() + \
                (item_bias ** 2).sum()
            embd_reg = (user_embeddings ** 2).sum() + \
                (item_embeddings ** 2).sum()
            reg = bias_reg * bias_weight_decay + embd_reg * embd_weight_decay

            (loss + reg).backward()

            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            pbar.set_description(f"loss: {np.mean(losses):.4f}")


def valid(model, metric, valid_dataloader):
    model = model.eval()

    targets = []
    outputs = []
    with torch.no_grad():
        with tqdm(valid_dataloader, dynamic_ncols=True) as pbar:
            for data in pbar:
                user_id = data["user_id"].cuda()
                item_id = data["item_id"].cuda()
                rating = data["rating"].cuda()

                output = model(user_id, item_id)[0]

                outputs.append(output.cpu())
                targets.append(rating.cpu())

    output = torch.cat(outputs)
    target = torch.cat(targets)
    loss = metric(output, target)

    log.info(f"RMSE loss: {loss.numpy():.4f}")


@hydra.main(config_path="conf", config_name="train")
def main(config):
    log.info(OmegaConf.to_yaml(config))
    if config.main.reproducibility.fix_rng:
        fix_rng(config.main.reproducibility.seed)


    df = MovieLense.get_df_cached(config.main.dataset)
    train_df, valid_df = split_df(df, [0.90, 0.10], shuffle=True)
    mu = train_df.rating.mean()
    user_id_converter = {user_id: idx + 1 for idx, user_id in enumerate(train_df.user_id.unique())}
    item_id_converter = {item_id: idx + 1 for idx, item_id in enumerate(train_df.item_id.unique())}
    num_users = len(user_id_converter)
    num_items = len(item_id_converter)

    model = Model(
        mu,
        num_users,
        num_items,
        num_factors=config.main.num_factors,
        dropout=config.main.dropout
    )
    model = model.cuda()

    train_dataset = Dataset(train_df, user_id_converter, item_id_converter)
    valid_dataset = Dataset(valid_df, user_id_converter, item_id_converter)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.main.batch_size, num_workers=32, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.main.batch_size, num_workers=32
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.main.lr,
        # weight_decay=config.main.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config.main.step_size, gamma=0.1,
        verbose=True,
    )

    criteria = MSE()
    metric = RMSE(
        min=config.main.min_rating,
        max=config.main.max_rating,
    )

    for epoch in range(config.main.epoch):
        log.info(f"epoch {epoch} start")
        train(
            model, optimizer, criteria, train_dataloader,
            bias_weight_decay=config.main.bias_weight_decay,
            embd_weight_decay=config.main.embd_weight_decay,
        )
        valid(model, metric, valid_dataloader, )
        scheduler.step()

    print("finished")


if __name__ == "__main__":
    main()
