from typing import Dict, List
import torch
import time
from torch.ao.pruning._experimental.data_sparsifier import DataNormSparsifier
import os
from dlrm_utils import get_dlrm_model, get_valid_name  # type: ignore[import]
import copy
import zipfile
from zipfile import ZipFile
import pandas as pd  # type: ignore[import]
import argparse


def create_attach_sparsifier(model, **sparse_config):
    """Create a DataNormSparsifier and the attach it to the model embedding layers

    Args:
        model (nn.Module)
            layer of the model that needs to be attached to the sparsifier
        sparse_config (Dict)
            Config to the DataNormSparsifier. Should contain the following keys:
                - sparse_block_shape
                - norm
                - sparsity_level
    """
    data_norm_sparsifier = DataNormSparsifier(**sparse_config)
    for name, parameter in model.named_parameters():
        if 'emb_l' in name:
            valid_name = get_valid_name(name)
            data_norm_sparsifier.add_data(name=valid_name, data=parameter)
    return data_norm_sparsifier


def save_model_states(state_dict, sparsified_model_dump_path, save_file_name, sparse_block_shape, norm, zip=True):
    """Dumps the state_dict() of the model.

    Args:
        state_dict (Dict)
            The state_dict() as dumped by dlrm_s_pytorch.py. Only the model state will be extracted
            from this dictionary. This corresponds to the 'state_dict' key in the state_dict dictionary.
            >>> model_state = state_dict['state_dict']
        save_file_name (str)
            The filename (not path) when saving the model state dictionary
        sparse_block_shape (Tuple)
            The block shape corresponding to the data norm sparsifier. **Used for creating save directory**
        norm (str)
            type of norm (L1, L2) for the datanorm sparsifier. **Used for creating save directory**
        zip (bool)
            if True, the file is zip-compressed.
    """
    folder_name = os.path.join(sparsified_model_dump_path, str(norm))

    # save model only states
    folder_str = f"config_{sparse_block_shape}"
    model_state = state_dict['state_dict']
    model_state_path = os.path.join(folder_name, folder_str, save_file_name)

    if not os.path.exists(os.path.dirname(model_state_path)):
        os.makedirs(os.path.dirname(model_state_path))
    torch.save(model_state, model_state_path)

    if zip:
        zip_path = model_state_path.replace('.ckpt', '.zip')
        with ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.write(model_state_path, save_file_name)
        os.remove(model_state_path)  # store it as zip, remove uncompressed
        model_state_path = zip_path

    model_state_path = os.path.abspath(model_state_path)
    file_size = os.path.getsize(model_state_path)
    file_size = file_size >> 20  # size in mb
    return model_state_path, file_size


def sparsify_model(path_to_model, sparsified_model_dump_path):
    """Sparsifies the embedding layers of the dlrm model for different sparsity levels, norms and block shapes
    using the DataNormSparsifier.
    The function tracks the step time of the sparsifier and the size of the compressed checkpoint and collates
    it into a csv.

    Note::
        This function dumps a csv sparse_model_metadata.csv in the current directory.

    Args:
        path_to_model (str)
            path to the trained criteo model ckpt file
        sparsity_levels (List of float)
            list of sparsity levels to be sparsified on
        norms (List of str)
            list of norms to be sparsified on
        sparse_block_shapes (List of tuples)
            List of sparse block shapes to be sparsified on
    """
    sparsity_levels = [sl / 10 for sl in range(0, 10)]
    sparsity_levels += [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

    norms = ["L1", "L2"]
    sparse_block_shapes = [(1, 1), (1, 4)]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Running for sparsity levels - ", sparsity_levels)
    print("Running for sparse block shapes - ", sparse_block_shapes)
    print("Running for norms - ", norms)

    orig_model = get_dlrm_model()
    saved_state = torch.load(path_to_model, map_location=device)
    orig_model.load_state_dict(saved_state['state_dict'])

    orig_model = orig_model.to(device)
    step_time_dict = {}

    stat_dict: Dict[str, List] = {'norm': [], 'sparse_block_shape': [], 'sparsity_level': [],
                                  'step_time_sec': [], 'zip_file_size': [], 'path': []}
    for norm in norms:
        for sbs in sparse_block_shapes:
            if norm == "L2" and sbs == (1, 1):
                continue
            for sl in sparsity_levels:
                model = copy.deepcopy(orig_model)
                sparsifier = create_attach_sparsifier(model, sparse_block_shape=sbs, norm=norm, sparsity_level=sl)

                t1 = time.time()
                sparsifier.step()
                t2 = time.time()

                step_time = t2 - t1
                norm_sl = f"{norm}_{sbs}_{sl}"
                print(f"Step Time for {norm_sl}=: {step_time} s")

                step_time_dict[norm_sl] = step_time

                sparsifier.squash_mask()

                saved_state['state_dict'] = model.state_dict()
                file_name = f'criteo_model_norm={norm}_sl={sl}.ckpt'
                state_path, file_size = save_model_states(saved_state, sparsified_model_dump_path, file_name, sbs, norm=norm)

                stat_dict['norm'].append(norm)
                stat_dict['sparse_block_shape'].append(sbs)
                stat_dict['sparsity_level'].append(sl)
                stat_dict['step_time_sec'].append(step_time)
                stat_dict['zip_file_size'].append(file_size)
                stat_dict['path'].append(state_path)

    df = pd.DataFrame(stat_dict)
    filename = 'sparse_model_metadata.csv'
    df.to_csv(filename, index=False)

    print(f"Saved sparsified metadata file in {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--sparsified_model_dump_path', type=str)
    args = parser.parse_args()

    sparsify_model(args.model_path, args.sparsified_model_dump_path)
