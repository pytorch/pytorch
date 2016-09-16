from .backend import FunctionBackend

class THNNFunctionBackend(FunctionBackend):

    def __reduce__(self):
        return (_get_thnn_function_backend, ())


def _get_thnn_function_backend():
    return backend


def _initialize_backend():
    from ..functions.thnn import _all_functions as _thnn_functions
    from ..functions.linear import Linear
    from ..functions.dropout import Dropout, FeatureDropout
    from ..functions.activation import Softsign
    from ..functions.loss import CosineEmbeddingLoss, \
            HingeEmbeddingLoss, MarginRankingLoss

    backend.register_function('Linear', Linear)
    backend.register_function('Dropout', Dropout)
    backend.register_function('Dropout2d', FeatureDropout)
    backend.register_function('Dropout3d', FeatureDropout)
    backend.register_function('CosineEmbeddingLoss', CosineEmbeddingLoss)
    backend.register_function('HingeEmbeddingLoss', HingeEmbeddingLoss)
    backend.register_function('MarginRankingLoss', MarginRankingLoss)
    backend.register_function('Softsign', Softsign)
    for cls in _thnn_functions:
        name = cls.__name__
        backend.register_function(name, cls)


backend = THNNFunctionBackend()
_initialize_backend()
