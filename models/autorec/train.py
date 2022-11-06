import logging

import hydra
import numpy as np
import scipy.sparse as ssp
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from utils.utils import fix_rng
from dataset.utils import split_df
from dataset.loader import MovieLense


log = logging.getLogger(__name__)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, mtx, dense=False, inp_dropout=0.0, oup_dropout=0.0):
        super().__init__()
        self.dense = dense
        data = mtx.data
        row, col = mtx.row, mtx.col

        mask = ssp.coo_array(([1] * len(data), (row, col)), shape=mtx.shape)
        mtx = mtx.tocsr()
        mask = mask.tocsr()
        if self.dense:
            mtx = mtx.todense()
            mask = mask.todense()
        self.mtx = mtx
        self.mask = mask

    def __len__(self):
        return self.mtx.shape[0] - 1  # user_id == 0

    def __getitem__(self, idx):
        data = self.mtx[[idx + 1], :]
        mask = self.mask[[idx + 1], :]
        if not self.dense:
            data = data.todense()
            mask = mask.todense()

        return {"data": data[0], "mask": mask[0]}


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, train_df, df, user_id_converter, item_id_converter):
        super().__init__()
        train_gdf = train_df.groupby("user_id").agg({"item_id": list, "rating": list})
        self.user_histories = dict()
        self.default_r = np.zeros(len(item_id_converter) + 1)
        for row in train_gdf.itertuples():
            c_item_ids = [item_id_converter[item_id] for item_id in row.item_id]
            r = np.zeros(len(item_id_converter) + 1)
            r[c_item_ids] = row.rating
            self.user_histories[row.Index] = r

        gdf = df.groupby("user_id").agg({"item_id": list, "rating": list})
        self.user_ids = []
        self.c_item_ids = []
        self.ratings = []
        self.is_unseens = []
        for row in gdf.itertuples():
            user_id = row.Index
            self.user_ids.append(user_id)
            self.c_item_ids.append([item_id_converter.get(item_id, 0) for item_id in row.item_id])
            self.ratings.append(row.rating)
            self.is_unseens.append([
                not (user_id in user_id_converter and item_id in item_id_converter)
                for item_id in row.item_id
            ])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        c_item_ids = self.c_item_ids[idx]
        ratings = self.ratings[idx]
        is_unseens = self.is_unseens[idx]
        user_history = self.user_histories.get(user_id, self.default_r.copy())

        return {"data": user_history, "ratings": ratings, "c_item_ids": c_item_ids, "is_unseens": is_unseens}

def valid_collate_fn(batch):
    return {
        "data": torch.tensor(np.array([i["data"] for i in batch])),
        "ratings": [i["ratings"] for i in batch],
        "c_item_ids": [i["c_item_ids"] for i in batch],
        "is_unseens": [i["is_unseens"] for i in batch],
    }

class Model(nn.Module):
    def __init__(self, num_input, f, g, k):
        super().__init__()
        self.num_input = num_input
        self.f = getattr(torch.nn, f)()
        self.g = getattr(torch.nn, g)()
        self.w = nn.Linear(k, num_input)
        self.v = nn.Linear(num_input, k)

    def forward(self, x, mask=None):
        assert mask is not None and self.training or mask is None and not self.training

        x = self.v(x)
        x = self.g(x)
        x = self.w(x)
        x = self.f(x)

        if self.training:
            nz = mask.nonzero()[:, 1]

            v_weight = self.v.weight[:, nz]
            w_weight = self.w.weight[nz]

            return x, v_weight, w_weight

        return x


class RMSE(nn.Module):
    def __init__(self, min=1., max=5.):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, output, target, mask=None):
        output = torch.clamp(output, min=self.min, max=self.max)
        if mask is not None:
            loss = torch.sum(((output - target) * mask)**2) / mask.sum()
        else:
            loss = torch.mean((output - target)**2)
        loss = torch.sqrt(loss)
        return loss


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        if mask is not None:
            loss = torch.sum(((output - target) * mask)**2)
            if mask.sum() != 0:
                loss = loss / mask.sum()
        else:
            loss = torch.mean((output - target)**2)
        return loss


def train(model, optimizer, criteria, train_dataloader, weight_decay=1.0):
    model = model.train()
    losses = []
    with tqdm(train_dataloader, dynamic_ncols=True) as pbar:
        for batch in pbar:
            data = batch["data"].float().cuda()
            mask = batch["mask"].float().cuda()

            output, v_weight, w_weight = model(data, mask)

            loss = criteria(output, data, mask)
            reg = weight_decay * ((v_weight ** 2).sum() + (w_weight ** 2).sum())

            optimizer.zero_grad()
            (loss + reg).backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            pbar.set_description(f"loss: {np.mean(losses):.4f}")


def valid(model, metric, valid_dataloader, unseen_r):
    model = model.eval()

    targets = []
    preds = []
    with torch.no_grad():
        with tqdm(valid_dataloader, dynamic_ncols=True) as pbar:
            for batch in pbar:
                data = batch["data"].float().cuda()
                ratings = batch["ratings"]
                c_item_ids = batch["c_item_ids"]
                is_unseens = batch["is_unseens"]

                outputs = model(data)

                for output, rating, c_item_id, is_unseen in zip(
                    outputs, ratings, c_item_ids, is_unseens
                ):
                    for _rating, _c_item_id, _is_unseen in zip(
                        rating, c_item_id, is_unseen
                    ):
                        targets.append(_rating)
                        preds.append(output[_c_item_id] if not _is_unseen else unseen_r)

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    loss = metric(preds, targets)

    log.info(f"RMSE loss: {loss.numpy():.4f}")


@hydra.main(config_path="conf", config_name="train")
def main(config):
    log.info(OmegaConf.to_yaml(config))
    if config.main.reproducibility.fix_rng:
        fix_rng(config.main.reproducibility.seed)

    df = MovieLense.get_df(config.main.dataset)
    if config.main.base == "item":
        user_id, item_id = df["user_id"].copy(), df["item_id"].copy()
        df["item_id"], df["user_id"] = user_id, item_id

    train_df, valid_df = split_df(df, [0.90, 0.10], shuffle=True)
    user_id_converter = {user_id: idx + 1 for idx, user_id in enumerate(train_df.user_id.unique())}
    item_id_converter = {item_id: idx + 1 for idx, item_id in enumerate(train_df.item_id.unique())}

    num_users = len(user_id_converter)
    num_items = len(item_id_converter)

    train_ratings, train_users, train_items = [], [], []
    for row in train_df.itertuples():
        train_users.append(user_id_converter[row.user_id])
        train_items.append(item_id_converter[row.item_id])
        train_ratings.append(row.rating)
    train_mtx = ssp.coo_array(
        (train_ratings, (train_users, train_items)),
        shape=(num_users + 1, num_items + 1)
    )

    train_dataset = TrainDataset(train_mtx)
    valid_dataset = ValidDataset(train_df, valid_df, user_id_converter, item_id_converter)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.main.batch_size, num_workers=32, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.main.val_batch_size, num_workers=32, collate_fn=valid_collate_fn
    )

    num_input = num_items
    model = Model(
        num_input + 1,
        config.main.f,
        config.main.g,
        config.main.k,
    )
    model = model.cuda()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.main.lr,
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
        train(model, optimizer, criteria, train_dataloader, weight_decay=config.main.weight_decay)
        valid(model, metric, valid_dataloader, unseen_r=config.main.unseen_r)
        scheduler.step()

    print("finished")


if __name__ == "__main__":
    main()
