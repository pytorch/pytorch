import argparse
import os
import pickle

import numpy as np
from torch._inductor.autotuner.model import AutotunerModel, ModelType

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--output_dir", type=str, default="./")
parser.add_argument("--full_train", action="store_true", default=False)


np.random.seed(0)
np.set_printoptions(threshold=np.inf, suppress=True)
np.set_printoptions(edgeitems=30, linewidth=100000)


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    full_train = args.full_train

    def load(file_name):
        file_name = os.path.join(data_dir, file_name)
        print("Loading " + file_name)
        with open(file_name, "rb") as f:
            return pickle.load(f)

    X_train = load("X_train.pkl")
    y_train = load("y_normalized_train.pkl")
    qid_train = load("qid_train.pkl")

    X_test = load("X_test.pkl")
    y_test = load("y_normalized_test.pkl")
    qid_test = load("qid_test.pkl")

    qid_train_unique = np.unique(qid_train)
    print(qid_train_unique[:10])

    qid_test_unique = np.unique(qid_test)
    print(qid_test_unique[:10])

    assert np.intersect1d(qid_train, qid_test).size == 0

    autotuner = AutotunerModel(ModelType.XGB_BASELINE)

    if full_train:
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
    else:
        X = X_train
        y = y_train

    autotuner.model.fit(
        X,
        y,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    qid_test_unique = np.unique(qid_test)
    for test_id in qid_test_unique[:10]:
        print(test_id)
        X_test = np.array(X_test)
        qid_test = np.array(qid_test)
        y_test = np.array(y_test)
        scores = autotuner.score_(X_test[qid_test == test_id])
        indices = np.argsort(scores)
        print(scores[indices])
        print(y_test[qid_test == test_id][indices])

    # dump model
    with open(os.path.join(output_dir, "xgb_baseline.pkl"), "wb") as f:
        pickle.dump(autotuner, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
