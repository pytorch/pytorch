# mypy: allow-untyped-defs
from typing import Dict, List
import torch
from dlrm_s_pytorch import unpack_batch  # type: ignore[import]
import numpy as np  # type: ignore[import]
import sklearn  # type: ignore[import]
from dlrm_utils import make_test_data_loader, dlrm_wrap, fetch_model  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import argparse


def inference_and_evaluation(dlrm, test_dataloader, device):
    """Perform inference and evaluation on the test dataset.
    The function returns the dictionary that contains evaluation metrics such as accuracy, f1, auc,
    precision, recall.
    Note: This function is a rewritten version of ```inference()``` present in dlrm_s_pytorch.py

    Args:
        dlrm (nn.Module)
            dlrm model object
        test_data_loader (torch dataloader):
            dataloader for the test dataset
        device (torch.device)
            device on which the inference happens
    """
    nbatches = len(test_dataloader)
    scores = []
    targets = []

    for i, testBatch in enumerate(test_dataloader):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break

        X_test, lS_o_test, lS_i_test, T_test, _, _ = unpack_batch(
            testBatch
        )
        # forward pass
        X_test, lS_o_test, lS_i_test = dlrm_wrap(X_test, lS_o_test, lS_i_test, device, ndevices=1)

        Z_test = dlrm(X_test, lS_o_test, lS_i_test)
        S_test = Z_test.detach().cpu().numpy()  # numpy array
        T_test = T_test.detach().cpu().numpy()  # numpy array
        scores.append(S_test)
        targets.append(T_test)

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)
    metrics = {
        "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "ap": sklearn.metrics.average_precision_score,
        "roc_auc": sklearn.metrics.roc_auc_score,
        "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "log_loss": lambda y_true, y_score: sklearn.metrics.log_loss(
            y_true=y_true, y_pred=y_score
        )
    }

    all_metrics = {}
    for metric_name, metric_function in metrics.items():
        all_metrics[metric_name] = round(metric_function(targets, scores), 3)

    return all_metrics


def evaluate_metrics(test_dataloader, sparse_model_metadata):
    """Evaluates the metrics the sparsified metrics for the dlrm model on various sparsity levels,
    block shapes and norms. This function evaluates the model on the test dataset and dumps
    evaluation metrics in a csv file [model_performance.csv]
    """
    metadata = pd.read_csv(sparse_model_metadata)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    metrics_dict: Dict[str, List] = {
        "norm": [],
        "sparse_block_shape": [],
        "sparsity_level": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "accuracy": [],
        "log_loss": []
    }

    for _, row in metadata.iterrows():
        norm, sbs, sl = row['norm'], row['sparse_block_shape'], row['sparsity_level']
        model_path = row['path']
        model = fetch_model(model_path, device)

        model_metrics = inference_and_evaluation(model, test_dataloader, device)
        key = f"{norm}_{sbs}_{sl}"
        print(key, "=", model_metrics)

        metrics_dict['norm'].append(norm)
        metrics_dict['sparse_block_shape'].append(sbs)
        metrics_dict['sparsity_level'].append(sl)

        for key, value in model_metrics.items():
            if key in metrics_dict:
                metrics_dict[key].append(value)

    sparse_model_metrics = pd.DataFrame(metrics_dict)
    print(sparse_model_metrics)

    filename = 'sparse_model_metrics.csv'
    sparse_model_metrics.to_csv(filename, index=False)
    print(f"Model metrics file saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-file', '--raw_data_file', type=str)
    parser.add_argument('--processed-data-file', '--processed_data_file', type=str)
    parser.add_argument('--sparse-model-metadata', '--sparse_model_metadata', type=str)

    args = parser.parse_args()

    # Fetch test data loader
    test_dataloader = make_test_data_loader(args.raw_data_file, args.processed_data_file)

    # Evaluate metrics
    evaluate_metrics(test_dataloader, args.sparse_model_metadata)
