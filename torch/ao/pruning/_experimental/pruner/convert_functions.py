import torch
from torch import nn
from torch.nn.utils import parametrize


def get_adjusted_next_layer_bias(next_layer, pruned_biases, mask):
    r"""Returns new adjusted bias for the second supported module"""
    if parametrize.is_parametrized(next_layer):
        # need to access original weight
        next_weight = next_layer.parametrizations.weight.original
    else:
        next_weight = next_layer.weight

    scaling_weight = next_weight[:, ~mask]
    if isinstance(next_layer, nn.Conv2d):  # checking for Conv2d
        # Propagating first layer pruned biases and calculating the new second layer bias
        # involves more steps since the Conv2d scaling weight has extra dimensions,
        # so adding bias involves broadcasting, logically:
        # for each channel k in range(oC):
        #     scaled_biases = sum(first_bias[pruned_idx] @ next_weight[k, pruned_idx, :, :].T)
        #     new_next_bias[k] = old_next_bias[k] + scaled_biases
        scaling_product = torch.matmul(
            pruned_biases.reshape(1, -1), torch.transpose(scaling_weight, 1, 2)
        )
        sum_range = list(range(len(scaling_product.shape)))[
            1:
        ]  # all but the first dimension
        scaled_biases = torch.sum(scaling_product, sum_range)
    elif isinstance(next_layer, nn.Linear):  # Linear
        scaled_biases = torch.matmul(
            pruned_biases, torch.transpose(scaling_weight, 0, 1)
        )  # recall b2_new = b1 @ w2.T + b2
    else:
        raise NotImplementedError(f"Type {type(next_layer)} not supported yet.")

    if (
        parametrize.is_parametrized(next_layer)
        and getattr(next_layer, "_bias", None) is not None
    ):  # next_layer is parametrized & has original bias ._bias
        adjusted_bias = nn.Parameter(scaled_biases + next_layer._bias)
    elif (
        not parametrize.is_parametrized(next_layer) and next_layer.bias is not None
    ):  # next_layer not parametrized & has .bias
        adjusted_bias = nn.Parameter(scaled_biases + next_layer.bias)
    else:  # next_layer has no bias
        adjusted_bias = nn.Parameter(scaled_biases)
    return adjusted_bias


def prune_module_bias(module, mask):
    # prune bias along with weights, discard pruned indices of bias
    original_bias = getattr(module, "_bias", module.bias)
    if original_bias is not None:
        module.bias = nn.Parameter(original_bias[mask])

    #  remove _bias parameter
    if hasattr(module, "_bias"):
        delattr(module, "_bias")


def propogate_module_bias(module, mask):
    # set current module bias
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias[mask])
    elif getattr(module, "_bias", None) is not None:
        module.bias = nn.Parameter(module._bias[mask])

    # get pruned biases to propogate to subsequent layer
    if getattr(module, "_bias", None) is not None:
        pruned_biases = module._bias[~mask]
    else:
        pruned_biases = None

    if hasattr(module, "_bias"):
        delattr(module, "_bias")

    return pruned_biases


# LINEAR CONVERSION FUNCTIONS
def convert_linear_helper(linear):
    mask = linear.parametrizations.weight[0].mask
    with torch.no_grad():
        parametrize.remove_parametrizations(linear, "weight", leave_parametrized=True)
        linear.weight = nn.Parameter(linear.weight[mask])
    linear.out_features = linear.weight.shape[0]
    return mask


def convert_linear(linear):
    mask = convert_linear_helper(linear)
    if getattr(linear, "prune_bias", False):
        prune_module_bias(linear, mask)


def convert_linear_linear(linear1, linear2):
    convert_linear_activation_linear(linear1, None, linear2)


def convert_linear_activation_linear(linear1, activation, linear2):
    mask = convert_linear_helper(linear1)
    if getattr(linear1, "prune_bias", False):
        prune_module_bias(linear1, mask)
    else:
        pruned_biases = propogate_module_bias(linear1, mask)
        if pruned_biases is not None:
            if activation:
                pruned_biases = activation(pruned_biases)
            linear2.bias = get_adjusted_next_layer_bias(linear2, pruned_biases, mask)

    with torch.no_grad():
        if parametrize.is_parametrized(linear2):
            linear2.parametrizations.weight.original = nn.Parameter(
                linear2.parametrizations.weight.original[:, mask]
            )
            linear2.in_features = linear2.parametrizations.weight.original.shape[1]
        else:
            linear2.weight = nn.Parameter(linear2.weight[:, mask])
            linear2.in_features = linear2.weight.shape[1]


# CONV2d CONVERSION FUNCTIONS
def convert_conv2d_helper(conv2d):
    mask = conv2d.parametrizations.weight[0].mask
    with torch.no_grad():
        parametrize.remove_parametrizations(conv2d, "weight", leave_parametrized=True)
        conv2d.weight = nn.Parameter(conv2d.weight[mask])
    conv2d.out_channels = conv2d.weight.shape[0]
    return mask


def convert_conv2d_padded(conv2d_1):
    # remove parameterization
    mask = conv2d_1.parametrizations.weight[0].mask
    with torch.no_grad():

        parametrize.remove_parametrizations(conv2d_1, "weight", leave_parametrized=True)

    if getattr(conv2d_1, "_bias", None) is not None:
        if (
            conv2d_1.bias is not None
        ):  # conv2d_1 has original bias and bias propagated from previous layer
            new_bias = torch.zeros(conv2d_1.bias.shape)
            new_bias[mask] = conv2d_1.bias[mask]
            # adjusted bias that to keep in conv2d_1
            new_bias[~mask] = conv2d_1._bias[~mask]
            # pruned biases that are kept instead of propagated
            conv2d_1.bias = nn.Parameter(new_bias)
        else:  # conv2d_1 has only original bias
            conv2d_1.bias = nn.Parameter(conv2d_1._bias)
    else:
        # no original bias, only propagated bias
        if (
            conv2d_1.bias is not None
        ):  # conv2d_1 has bias propagated from previous layer
            conv2d_1.bias.data[~mask] = 0

    if hasattr(conv2d_1, "_bias"):
        delattr(conv2d_1, "_bias")


def convert_conv2d(conv2d):
    mask = convert_conv2d_helper(conv2d)
    if getattr(conv2d, "prune_bias", False):
        prune_module_bias(conv2d, mask)


def convert_conv2d_conv2d(conv2d_1, conv2d_2):
    convert_conv2d_activation_conv2d(conv2d_1, None, conv2d_2)


def convert_conv2d_activation_conv2d(conv2d_1, activation, conv2d_2):
    r"""
    Fusion Pattern for conv2d -> some activation module / function -> conv2d layers
    """
    mask = conv2d_1.parametrizations.weight[0].mask
    prune_bias = getattr(conv2d_1, "prune_bias", False)
    if (
        hasattr(conv2d_2, "padding")
        and conv2d_2.padding > (0, 0)
        and (conv2d_1.bias is not None or getattr(conv2d_1, "_bias", None) is not None)
    ):
        convert_conv2d_padded(conv2d_1)
    else:
        mask = convert_conv2d_helper(conv2d_1)
        if prune_bias:
            prune_module_bias(conv2d_1, mask)
        else:
            pruned_biases = propogate_module_bias(conv2d_1, mask)
            if pruned_biases is not None:
                if activation:
                    pruned_biases = activation(pruned_biases)
                conv2d_2.bias = get_adjusted_next_layer_bias(
                    conv2d_2, pruned_biases, mask
                )

        if (
            not (hasattr(conv2d_2, "padding") and conv2d_2.padding > (0, 0))
            or conv2d_1.bias is None
        ):
            with torch.no_grad():
                if parametrize.is_parametrized(conv2d_2):
                    conv2d_2.parametrizations.weight.original = nn.Parameter(
                        conv2d_2.parametrizations.weight.original[:, mask]
                    )
                    conv2d_2.in_channels = (
                        conv2d_2.parametrizations.weight.original.shape[1]
                    )
                else:
                    conv2d_2.weight = nn.Parameter(conv2d_2.weight[:, mask])
                    conv2d_2.in_channels = conv2d_2.weight.shape[1]


def convert_conv2d_pool_activation_conv2d(c1, pool, activation, c2):
    convert_conv2d_activation_conv2d(c1, activation, c2)


def convert_conv2d_activation_pool_conv2d(c1, activation, pool, c2):
    convert_conv2d_activation_conv2d(c1, activation, c2)


def convert_conv2d_pool_flatten_linear(conv2d, pool, flatten, linear):
    mask = convert_conv2d_helper(conv2d)

    # We map the pruned indices of the Conv2d output to the flattened indices of the Linear following the Flatten layer.
    # we determine the flattening scale (h * w), and readjust `first_pruned_indices`
    # (each idx maps to range idx * h * w to (idx+1) * h * w), `first_valid_indices`,
    # and `pruned_biases` (repeat each bias by h * w).
    if parametrize.is_parametrized(linear):
        linear_ic = linear.parametrizations.weight.original.shape[1]
    else:
        linear_ic = linear.weight.shape[1]
    conv2d_oc = len(mask)

    assert (
        linear_ic % conv2d_oc == 0
    ), f"Flattening from dimensions {conv2d_oc} to {linear_ic} not supported"

    flatten_scale = linear_ic // conv2d_oc
    flattened_mask = torch.tensor(
        [[val] * flatten_scale for val in mask], dtype=torch.bool
    ).flatten()

    if getattr(conv2d, "prune_bias", False):
        prune_module_bias(conv2d, mask)
    else:
        pruned_biases = propogate_module_bias(conv2d, mask)
        flattened_pruned_biases = torch.tensor(
            [[bias] * flatten_scale for bias in pruned_biases]
        ).flatten()
        linear.bias = get_adjusted_next_layer_bias(
            linear, flattened_pruned_biases, flattened_mask
        )

    with torch.no_grad():
        if parametrize.is_parametrized(linear):
            linear.parametrizations.weight.original = nn.Parameter(
                linear.parametrizations.weight.original[:, flattened_mask]
            )
            linear.in_features = linear.parametrizations.weight.original.shape[1]
        else:
            linear.weight = nn.Parameter(linear.weight[:, flattened_mask])
            linear.in_features = linear.weight.shape[1]
