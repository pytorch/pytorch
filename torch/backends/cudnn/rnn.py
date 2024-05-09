import torch.cuda

try:
    from torch._C import _cudnn
except ImportError:
    # Uses of all the functions below should be guarded by torch.backends.cudnn.is_available(),
    # so it's safe to not emit any checks here.
    _cudnn = None  # type: ignore[assignment]


def get_cudnn_mode(mode):
    if mode == "RNN_RELU":
        return int(_cudnn.RNNMode.rnn_relu)
    elif mode == "RNN_TANH":
        return int(_cudnn.RNNMode.rnn_tanh)
    elif mode == "LSTM":
        return int(_cudnn.RNNMode.lstm)
    elif mode == "GRU":
        return int(_cudnn.RNNMode.gru)
    else:
        raise Exception(f"Unknown mode: {mode}")  # noqa: TRY002


# NB: We don't actually need this class anymore (in fact, we could serialize the
# dropout state for even better reproducibility), but it is kept for backwards
# compatibility for old models.
class Unserializable:
    def __init__(self, inner):
        self.inner = inner

    def get(self):
        return self.inner

    def __getstate__(self):
        # Note: can't return {}, because python2 won't call __setstate__
        # if the value evaluates to False
        return "<unserializable>"

    def __setstate__(self, state):
        self.inner = None


def init_dropout_state(dropout, train, dropout_seed, dropout_state):
    dropout_desc_name = "desc_" + str(torch.cuda.current_device())
    dropout_p = dropout if train else 0
    if (dropout_desc_name not in dropout_state) or (
        dropout_state[dropout_desc_name].get() is None
    ):
        if dropout_p == 0:
            dropout_state[dropout_desc_name] = Unserializable(None)
        else:
            dropout_state[dropout_desc_name] = Unserializable(
                torch._cudnn_init_dropout_state(  # type: ignore[call-arg]
                    dropout_p,
                    train,
                    dropout_seed,
                    self_ty=torch.uint8,
                    device=torch.device("cuda"),
                )
            )
    dropout_ts = dropout_state[dropout_desc_name].get()
    return dropout_ts
