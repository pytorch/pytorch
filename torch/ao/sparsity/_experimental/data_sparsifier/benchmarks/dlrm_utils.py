
from dlrm_s_pytorch import DLRM_Net  # type: ignore[import]
import numpy as np  # type: ignore[import]


def get_valid_name(name):
    """Replaces '.' with '_' as names with '.' are invalid in data sparsifier
    """
    return name.replace('.', '_')


def get_dlrm_model():
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
    dlrm_model = DLRM_Net(**dlrm_model_config)
    return dlrm_model
