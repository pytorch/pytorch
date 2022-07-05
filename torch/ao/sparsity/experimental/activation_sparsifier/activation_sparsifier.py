from typing import Dict, Any
from collections import defaultdict
from torch import nn

__all__ = ['ActivationSparsifier']


class ActivationSparsifier:
    r"""
    The Activation sparsifier class aims to sparsify/prune activations in a neural
    network. The idea is to attach the sparsifier to a layer (or layers) and it
    zeroes out the activations based on the mask_fn (or sparsification function)
    input by the user.
    The mask_fn is applied once all the inputs are aggregated and reduced i.e.
    mask = mask_fn(reduce_fn(aggregate_fn(activations)))

    Note::
        The sparsification mask is computed on the input **before it goes through the attached layer**.

    Args:
        model (nn.Module)
            The model whose layers will be sparsified. The layers that needs to be
            sparsified should be added separately using the register_layer() function

        aggregate_fn (Optional, function type)
            default aggregate_fn that is used if not specified while registering the layer.
            specifies how inputs should be aggreagated over time.

        reduce_fn (Optional, function type)
            default reduce_fn that is used if not specified while registering the layer.
            reduce_fn will be applied on the aggregated input

        mask_fn (Optional, function type)
            default mask_fn that is used to create the sparsification mask (based on the
            aggregated & reduced input). This is used if not specified while registering the layer

        features (Optional, list)
            default selected features to sparsify.
            If this is non-empty, then the mask_fn will be applied for each feature of the input.
            For example,
            >>> mask = [mask_fn(reduce_fn(aggregated_fn(input[feature])) for feature in features]

        feature_dim (Optional, int)
            default dimension of input features. Again, features along this dim will be chosen
            for sparsification.

        sparse_config (Dict)
            Default configuration for the mask_fn. This config will be passed
            with the mask_fn()

    Expected Usage:
        >>> model = SomeModel()
        >>> act_sparsifier = ActivationSparsifier(...)  # init activation sparsifier

        >>> # Initialize aggregate_fn
        >>> def agg_fn(x, y):
        >>>     return x + y

        >>> # Initialize reduce_fn
        >>> def reduce_fn(x):
        >>>     return torch.mean(x, dim=0)

        >>> # Initialize mask_fn
        >>> def mask_fn(data):
        >>>     return torch.eye(data.shape).to(data.device)


        >>> act_sparsifier.register_layer(model.some_layer, aggregate_fn=agg_fn, reduce_fn=reduce_fn, mask_fn=mask_fn)

        >>> # start training process
            >>> # epoch starts
                >>> # model.forward(), compute_loss() and model.backwards()
            >>> # epoch ends
            >>> act_sparsifier.step()
        >>> # end training process

        >>> sparsifier.squash_mask()
    """
    def __init__(self, model: nn.Module, aggregate_fn=None, reduce_fn=None, mask_fn=None,
                 features=None, feature_dim=None, **sparse_config):
        self.model = model
        self.defaults: Dict[str, Any] = defaultdict()
        self.defaults['sparse_config'] = sparse_config

        # functions
        self.defaults['aggregate_fn'] = aggregate_fn
        self.defaults['reduce_fn'] = reduce_fn
        self.defaults['mask_fn'] = mask_fn

        # default feature and feature_dim
        self.defaults['features'] = features
        self.defaults['feature_dim'] = feature_dim

        self.data_groups: Dict[str, Dict] = defaultdict(dict)  # contains all relevant info w.r.t each registered layer

        self.state: Dict[str, Any] = defaultdict(dict)  # layer name -> mask

    def register_layer(self, layer: nn.Module, aggregate_fn=None, reduce_fn=None,
                       mask_fn=None, features=None, feature_dim=None, **config):
        r"""
        Registers a layer for sparsification. The layer should be part of self.model.
        Specifically, registers a pre-forward hook to the layer. The hook will apply the aggregate_fn
        and store the aggregated activations that is input over each step.

        Optionally, accepts custom aggregate, reduce, create_mask functions and features, dimension.

        Note::
            There is no need to pass in the name of the layer as it is automatically computed as per
            the fqn convention.

            All the functions (fn) passed as argument will be called at a dim, feature level.
        """
        pass

    def get_mask(self, name: str):
        """
        Returns mask associated to the layer.

        The mask is
            - a torch tensor is features for that layer is None.
            - a list of torch tensors for each feature, otherwise

        Note::
            The shape of the mask is unknown until model.forward() is applied.
            Hence, if get_mask() is called before model.forward(), then an
            error will be raised.
        """
        pass

    def step(self):
        """
        step() does the following for each registered layer -
            1. apply reduce_fn on the aggregated activations
            2. use mask_fn to compute the sparsification mask
        """
        pass

    def update_mask(self, name, data, **config):
        """Computes and updates mask based on config, feature and dim and data
        """
        pass

    def squash_mask(self, **kwargs):
        """
        Unregisters aggreagate hook that was applied earlier and registers sparsification hooks.
        The sparsification hook will apply the mask to the activations before it is fed into the
        attached layer.
        """
        pass
