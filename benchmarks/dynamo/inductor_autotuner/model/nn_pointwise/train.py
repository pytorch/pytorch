import argparse
import tqdm
import os
import pickle
import numpy as np
import torch
from torch._inductor.autotuner.model import AutotunerModel, ModelType

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--output_dir", type=str, default="./")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--full_train", action="store_true", default=False)


np.random.seed(0)
np.set_printoptions(threshold=np.inf, suppress=True)
np.set_printoptions(edgeitems=30, linewidth=100000)


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    n_epoch = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    full_train = args.full_train

    def load(file_name):
        file_name = os.path.join(data_dir, file_name)
        print("Loading " + file_name)
        with open(file_name, "rb") as f:
            return pickle.load(f)

    X_train = load("X_train.pkl")
    y_train = load("y_normalized_train.pkl")

    X_test = load("X_test.pkl")
    y_test = load("y_normalized_test.pkl")

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    autotuner = AutotunerModel(ModelType.NN_POINTWISE)

    if full_train:
        X = tuple(
            torch.concatenate([X_train[i], X_test[i]]) for i in range(len(X_train))
        )
        y = np.concatenate([y_train, y_test])
    else:
        X = X_train
        y = y_train

    def get_loss(X_loss, y_loss):
        test_batch_size = 8192 * 8
        mse_loss_sum = 0
        mae_loss_sum = 0
        autotuner.model.eval()

        with torch.no_grad():
            for i in range(0, y_loss.shape[0], test_batch_size):
                X_loss_batch = tuple(
                    X_group[i : i + test_batch_size].to("cuda") for X_group in X_loss
                )
                y_pred = autotuner.model.forward_(X_loss_batch)
                mse_loss = torch.nn.functional.mse_loss(
                    y_pred.squeeze(),
                    torch.from_numpy(y_loss[i : i + test_batch_size]).to("cuda"),
                )
                mae_loss = torch.nn.functional.l1_loss(
                    y_pred.squeeze(),
                    torch.from_numpy(y_loss[i : i + test_batch_size]).to("cuda"),
                )

                mse_loss_sum += mse_loss.item() * y_pred.shape[0]
                mae_loss_sum += mae_loss.item() * y_pred.shape[0]
            torch.cuda.empty_cache()

        return mse_loss_sum / y_loss.shape[0], mae_loss_sum / y_loss.shape[0]

    autotuner.model.to("cuda")
    optimizer = torch.optim.Adam(autotuner.model.parameters(), lr=lr)
    print("X")
    for X_group in X:
        print(X_group.shape)
    print("y", y.shape)

    for epoch in range(n_epoch):
        autotuner.model.train()

        permutation = np.random.permutation(y.shape[0])
        X_epoch = tuple(X_group[permutation] for X_group in X)
        y_epoch = y[permutation]

        train_mse_loss = 0
        train_mae_loss = 0

        for i in tqdm.tqdm(range(0, y_epoch.shape[0], batch_size)):
            optimizer.zero_grad()
            X_tuple_batch = tuple(
                X_group[i : i + batch_size].to("cuda") for X_group in X_epoch
            )
            y_batch = torch.from_numpy(y_epoch[i : i + batch_size]).to("cuda").float()
            y_pred = autotuner.model.forward_(X_tuple_batch)

            loss = torch.nn.functional.mse_loss(
                y_pred.squeeze(), y_batch, reduction="mean"
            )
            loss_mae = torch.nn.functional.l1_loss(
                y_pred.squeeze(), y_batch, reduction="mean"
            )

            train_mse_loss += loss.item() * y_pred.shape[0]
            train_mae_loss += loss_mae.item() * y_pred.shape[0]

            loss.backward()
            optimizer.step()

        print(
            f"Train: epoch={epoch} rmse = {np.sqrt(train_mse_loss / y.shape[0])}, mae = {train_mae_loss / y.shape[0]}",
            end=" ||| ",
        )

        mse_loss, mae_loss = get_loss(X_test, y_test)
        print(f"Test: epoch={epoch} rmse = {np.sqrt(mse_loss)}, mae = {mae_loss}")

        # dump model
        with open(
            os.path.join(
                output_dir,
                f"nn_pointwise_{full_train}_{epoch}_{np.sqrt(mse_loss)}_{mae_loss}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(autotuner, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
