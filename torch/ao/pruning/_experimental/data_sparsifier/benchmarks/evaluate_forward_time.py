# mypy: allow-untyped-defs
from typing import Dict, List
import torch
from dlrm_s_pytorch import unpack_batch  # type: ignore[import]
import numpy as np  # type: ignore[import]
import time
from dlrm_utils import make_test_data_loader, fetch_model, dlrm_wrap  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import argparse


def run_forward(model, **batch):
    """The purpose of this function is to time the forward run of the model.
    The model forward happens a 100 times and each pass is timed. The average
    of this 100 runs is returned as avg_time.
    """
    time_list = []
    X, lS_o, lS_i = batch['X'], batch['lS_o'], batch['lS_i']
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            model(X, lS_o, lS_i)
        end = time.time()
        time_taken = end - start
        time_list.append(time_taken)
    avg_time = np.mean(time_list[1:])
    return avg_time


def make_sample_test_batch(raw_data_path, processed_data_path, device):
    """Create the test_data_loader and sample a batch from it. This batch will be used
    to measure the forward pass of the model throughout this experiment.
    """
    test_data_loader = make_test_data_loader(raw_data_path, processed_data_path)

    test_iter = iter(test_data_loader)

    test_batch = next(test_iter)

    X_test, lS_o_test, lS_i_test, _, _, _ = unpack_batch(test_batch)

    X, lS_o, lS_i = dlrm_wrap(X_test, lS_o_test, lS_i_test, device)
    batch = {
        'X': X,
        'lS_o': lS_o,
        'lS_i': lS_i
    }

    return batch

def measure_forward_pass(sparse_model_metadata, device, sparse_dlrm, **batch):
    """Measures and tracks the forward pass of the model for all the sparsity levels, block shapes and norms
    available in sparse_model_metadata file.
    If sparse_dlrm=True, then the SparseDLRM model is loaded, otherwise the standard one is.
    """
    time_taken_dict: Dict[str, List] = {
        "norm": [],
        "sparse_block_shape": [],
        "sparsity_level": [],
        "time_taken": [],
    }

    metadata = pd.read_csv(sparse_model_metadata)

    for _, row in metadata.iterrows():
        norm, sbs, sl = row['norm'], row['sparse_block_shape'], row['sparsity_level']
        model_path = row['path']
        model = fetch_model(model_path, device, sparse_dlrm=sparse_dlrm)
        time_taken = run_forward(model, **batch)
        out_str = f"{norm}_{sbs}_{sl}={time_taken}"
        print(out_str)
        time_taken_dict["norm"].append(norm)
        time_taken_dict["sparse_block_shape"].append(sbs)
        time_taken_dict["sparsity_level"].append(sl)
        time_taken_dict["time_taken"].append(time_taken)

    time_df = pd.DataFrame(time_taken_dict)

    if sparse_dlrm:
        time_df['dlrm_type'] = 'with_torch_sparse'
    else:
        time_df['dlrm_type'] = 'without_torch_sparse'

    return time_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-file', '--raw_data_file', type=str)
    parser.add_argument('--processed-data-file', '--processed_data_file', type=str)
    parser.add_argument('--sparse-model-metadata', '--sparse_model_metadata', type=str)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    batch = make_sample_test_batch(args.raw_data_file, args.processed_data_file, device)

    print("Forward Time for Sparse DLRM")
    sparse_dlrm_time_df = measure_forward_pass(args.sparse_model_metadata, device, sparse_dlrm=True, **batch)
    print(sparse_dlrm_time_df)

    print("Forward Time for Normal DLRM")
    norm_dlrm_time_df = measure_forward_pass(args.sparse_model_metadata, device, sparse_dlrm=False, **batch)
    print(norm_dlrm_time_df)

    forward_time_all = pd.concat([sparse_dlrm_time_df, norm_dlrm_time_df])
    forward_time_all.to_csv('dlrm_forward_time_info.csv', index=False)
