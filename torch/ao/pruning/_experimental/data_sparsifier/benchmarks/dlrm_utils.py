# mypy: allow-untyped-defs
import torch
from dlrm_s_pytorch import DLRM_Net  # type: ignore[import]
import numpy as np  # type: ignore[import]
from dlrm_data_pytorch import CriteoDataset, collate_wrapper_criteo_offset  # type: ignore[import]
import zipfile
import os


class SparseDLRM(DLRM_Net):
    """The SparseDLRM model is a wrapper around the DLRM_Net model that tries
    to use torch.sparse tensors for the features obtained after the ```interact_features()```
    call. The idea is to do a simple torch.mm() with the weight matrix of the first linear
    layer of the top layer.
    """
    def __init__(self, **args):
        super().__init__(**args)

    def forward(self, dense_x, lS_o, lS_i):
        x = self.apply_mlp(dense_x, self.bot_l)  # dense features
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)  # apply embedding bag
        z = self.interact_features(x, ly)

        z = z.to_sparse_coo()
        z = torch.mm(z, self.top_l[0].weight.T).add(self.top_l[0].bias)
        for layer in self.top_l[1:]:
            z = layer(z)

        return z


def get_valid_name(name):
    """Replaces '.' with '_' as names with '.' are invalid in data sparsifier
    """
    return name.replace('.', '_')


def get_dlrm_model(sparse_dlrm=False):
    """Obtain dlrm model. The configs specified are based on the script in
    bench/dlrm_s_criteo_kaggle.sh. The same config is used to train the model
    for benchmarking on data sparsifier.
    """
    dlrm_model_config = {
        'm_spa': 16,
        'ln_emb': np.array([1460, 583, 10131227, 2202608, 305, 24,
                            12517, 633, 3, 93145, 5683, 8351593,
                            3194, 27, 14992, 5461306, 10, 5652,
                            2173, 4, 7046547, 18, 15, 286181,
                            105, 142572], dtype=np.int32),
        'ln_bot': np.array([13, 512, 256, 64, 16]),
        'ln_top': np.array([367, 512, 256, 1]),
        'arch_interaction_op': 'dot',
        'arch_interaction_itself': False,
        'sigmoid_bot': -1,
        'sigmoid_top': 2,
        'sync_dense_params': True,
        'loss_threshold': 0.0,
        'ndevices': 1,
        'qr_flag': False,
        'qr_operation': 'mult',
        'qr_collisions': 4,
        'qr_threshold': 200,
        'md_flag': False,
        'md_threshold': 200,
        'weighted_pooling': None,
        'loss_function': 'bce'
    }
    if sparse_dlrm:
        dlrm_model = SparseDLRM(**dlrm_model_config)
    else:
        dlrm_model = DLRM_Net(**dlrm_model_config)
    return dlrm_model


def dlrm_wrap(X, lS_o, lS_i, device, ndevices=1):
    """Rewritten simpler version of ```dlrm_wrap()``` found in dlrm_s_pytorch.py.
    This function simply moves the input tensors into the device and without the forward pass
    """
    if ndevices == 1:
        lS_i = (
            [S_i.to(device) for S_i in lS_i]
            if isinstance(lS_i, list)
            else lS_i.to(device)
        )
        lS_o = (
            [S_o.to(device) for S_o in lS_o]
            if isinstance(lS_o, list)
            else lS_o.to(device)
        )
    return X.to(device), lS_o, lS_i


def make_test_data_loader(raw_data_file_path, processed_data_file):
    """Function to create dataset and dataloaders for the test dataset.
    Rewritten simpler version of ```make_criteo_and_loaders()``` from the dlrm_data_pytorch.py
    that makes the test dataset and dataloaders only for the ***kaggle criteo dataset***
    """
    test_data = CriteoDataset(
        "kaggle",
        -1,
        0.0,
        "total",
        "test",
        raw_data_file_path,
        processed_data_file,
        False,
        False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=16384,
        shuffle=False,
        num_workers=7,
        collate_fn=collate_wrapper_criteo_offset,
        pin_memory=False,
        drop_last=False,
    )
    return test_loader


def fetch_model(model_path, device, sparse_dlrm=False):
    """This function unzips the zipped model checkpoint (if zipped) and returns a
    model object

    Args:
        model_path (str)
            path pointing to the zipped/raw model checkpoint file that was dumped in evaluate disk savings
        device (torch.device)
            device to which model needs to be loaded to
    """
    if zipfile.is_zipfile(model_path):
        with zipfile.ZipFile(model_path, 'r', zipfile.ZIP_DEFLATED) as zip_ref:
            zip_ref.extractall(os.path.dirname(model_path))
            unzip_path = model_path.replace('.zip', '.ckpt')
    else:
        unzip_path = model_path

    model = get_dlrm_model(sparse_dlrm=sparse_dlrm)
    model.load_state_dict(torch.load(unzip_path, map_location=device))
    model = model.to(device)
    model.eval()

    # If there was a zip file, clean up the unzipped files
    if zipfile.is_zipfile(model_path):
        os.remove(unzip_path)

    return model
