import argparse
import os
import pickle

import numpy as np
from torch._inductor.autotuner.model import ModelType

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="./")
parser.add_argument("--model-dir", type=str, default=None)
parser.add_argument("--model-name", type=str, required=True)


np.random.seed(0)
np.set_printoptions(threshold=np.inf, suppress=True)
np.set_printoptions(edgeitems=30, linewidth=100000)


def main(args):
    data_dir = args.data_dir
    model_dir = args.model_dir
    model_name = args.model_name
    if model_dir is None:
        model_dir = data_dir

    def load(file_name, dir=data_dir):
        file_name = os.path.join(dir, file_name)
        print("Loading " + file_name)
        with open(file_name, "rb") as f:
            return pickle.load(f)

    x_train = load("X_train.pkl")
    y_train = load("y_train.pkl")
    y_baseline_train = load("y_baseline_train.pkl")
    qid_train = load("qid_train.pkl")

    x_test = load("X_test.pkl")
    y_test = load("y_test.pkl")
    y_baseline_test = load("y_baseline_test.pkl")
    qid_test = load("qid_test.pkl")

    qid_train_unique = np.unique(qid_train)
    print(qid_train_unique[:10])

    qid_test_unique = np.unique(qid_test)
    print(qid_test_unique[:10])

    assert np.intersect1d(qid_train, qid_test).size == 0

    autotuner = load(model_name, dir=model_dir)

    def measure(X, y, y_baseline, qid):
        qid_unique = np.unique(qid)
        pointer = 0

        acc_top1 = 0
        acc_top2 = 0
        acc_top5 = 0
        gap_top1 = 0
        gap_top2 = 0
        gap_top5 = 0
        gap_true = 0
        counter = 0

        for i, test_id in enumerate(qid_unique):
            if autotuner.model_type == ModelType.XGB_BASELINE:
                x_group = list()
                y_group = list()
                while pointer < len(qid) and qid[pointer] == test_id:
                    x_group.append(X[pointer])
                    y_group.append(y[pointer])
                    y_baseline_ = y_baseline[pointer]
                    pointer += 1
                scores = autotuner.score_(x_group)
            else:
                x_group = list()
                y_group = list()
                while pointer < len(qid) and qid[pointer] == test_id:
                    x_group.append(pointer)
                    y_group.append(y[pointer])
                    y_baseline_ = y_baseline[pointer]
                    pointer += 1
                x_group = np.array(x_group)
                x_group = tuple(Xg[x_group].to("cuda") for Xg in X)
                autotuner.model.eval()
                scores = (
                    autotuner.model.forward_(x_group).squeeze().cpu().detach().numpy()
                )
                if autotuner.model_type == ModelType.NN_POINTWISE:
                    scores = scores * -1

            y_group = np.array(y_group)
            y_pred = y_group[np.argsort(scores)]
            y_pred_top1 = y_pred[:1].min()
            y_pred_top2 = y_pred[:2].min()
            y_pred_top5 = y_pred[:5].min()

            y_true = min(y_group)

            acc_top1 += y_pred_top1 <= y_baseline_
            acc_top2 += y_pred_top2 <= y_baseline_
            acc_top5 += y_pred_top5 <= y_baseline_
            gap_top1 += y_pred_top1 - y_baseline_
            gap_top2 += y_pred_top2 - y_baseline_
            gap_top5 += y_pred_top5 - y_baseline_
            gap_true += y_true - y_baseline_

            if y_pred_top1 - y_baseline_ > 0.05:
                counter += 1
                print("test_id:", test_id)
                print("y_group", y_group)
                print("y_pred", y_pred)
                print("y_true", y_true)
                print("y_baseline", y_baseline_)
                print("y_pred_top1", y_pred_top1, "->", y_pred_top1 - y_baseline_)
                print("y_pred_top2", y_pred_top2, "->", y_pred_top2 - y_baseline_)
                print("y_pred_top5", y_pred_top5, "->", y_pred_top5 - y_baseline_)
                print("counter", counter, "ratio", counter / (i + 1) * 100)

        print("acc_top1", acc_top1 / len(qid_unique) * 100)
        print("acc_top2", acc_top2 / len(qid_unique) * 100)
        print("acc_top5", acc_top5 / len(qid_unique) * 100)
        print("gap_top1", gap_top1 / len(qid_unique))
        print("gap_top2", gap_top2 / len(qid_unique))
        print("gap_top5", gap_top5 / len(qid_unique))
        print("gap_true", gap_true / len(qid_unique))
        print("counter", counter, "ratio", counter / len(qid_unique) * 100)

    print("###################################################### Training set:")
    measure(x_train, y_train, y_baseline_train, qid_train)
    print("###################################################### Testing set:")
    measure(x_test, y_test, y_baseline_test, qid_test)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
