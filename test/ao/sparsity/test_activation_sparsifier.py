# Owner(s): ["module: sparse"]

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.pruning._experimental.activation_sparsifier.activation_sparsifier import (
    ActivationSparsifier,
)
from torch.ao.pruning.sparsifier.utils import module_to_fqn
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfTorchDynamo,
    TestCase,
)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.identity1 = nn.Identity()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(4608, 128)
        self.identity2 = nn.Identity()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.identity1(out)
        out = self.max_pool1(out)

        batch_size = x.shape[0]
        out = out.reshape(batch_size, -1)

        out = F.relu(self.identity2(self.linear1(out)))
        out = self.linear2(out)
        return out


class TestActivationSparsifier(TestCase):
    def _check_constructor(self, activation_sparsifier, model, defaults, sparse_config):
        """Helper function to check if the model, defaults and sparse_config are loaded correctly
        in the activation sparsifier
        """
        sparsifier_defaults = activation_sparsifier.defaults
        combined_defaults = {**defaults, "sparse_config": sparse_config}

        # more keys are populated in activation sparsifier (even though they may be None)
        if len(combined_defaults) > len(activation_sparsifier.defaults):
            raise AssertionError(
                f"Expected combined_defaults length <= sparsifier.defaults length, "
                f"got {len(combined_defaults)} > {len(activation_sparsifier.defaults)}"
            )

        for key, config in sparsifier_defaults.items():
            # all the keys in combined_defaults should be present in sparsifier defaults
            if config != combined_defaults.get(key):
                raise AssertionError(
                    f"Expected sparsifier_defaults['{key}'] == combined_defaults.get('{key}'), "
                    f"got {config} != {combined_defaults.get(key)}"
                )

    def _check_register_layer(
        self, activation_sparsifier, defaults, sparse_config, layer_args_list
    ):
        """Checks if layers in the model are correctly mapped to it's arguments.

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            defaults (Dict)
                all default config (except sparse_config)

            sparse_config (Dict)
                default sparse config passed to the sparsifier

            layer_args_list (list of tuples)
                Each entry in the list corresponds to the layer arguments.
                First entry in the tuple corresponds to all the arguments other than sparse_config
                Second entry in the tuple corresponds to sparse_config
        """
        # check args
        data_groups = activation_sparsifier.data_groups
        if len(data_groups) != len(layer_args_list):
            raise AssertionError(
                f"Expected {len(layer_args_list)} data_groups, got {len(data_groups)}"
            )
        for layer_args in layer_args_list:
            layer_arg, sparse_config_layer = layer_args

            # check sparse config
            sparse_config_actual = copy.deepcopy(sparse_config)
            sparse_config_actual.update(sparse_config_layer)

            name = module_to_fqn(activation_sparsifier.model, layer_arg["layer"])

            if data_groups[name]["sparse_config"] != sparse_config_actual:
                raise AssertionError(
                    f"Expected sparse_config {sparse_config_actual}, "
                    f"got {data_groups[name]['sparse_config']}"
                )

            # assert the rest
            other_config_actual = copy.deepcopy(defaults)
            other_config_actual.update(layer_arg)
            other_config_actual.pop("layer")

            for key, value in other_config_actual.items():
                if key not in data_groups[name]:
                    raise AssertionError(
                        f"Expected key '{key}' in data_groups['{name}']"
                    )
                if value != data_groups[name][key]:
                    raise AssertionError(
                        f"Expected data_groups['{name}']['{key}'] == {value}, got {data_groups[name][key]}"
                    )

            # get_mask should raise error
            with self.assertRaises(ValueError):
                activation_sparsifier.get_mask(name=name)

    def _check_pre_forward_hook(self, activation_sparsifier, data_list):
        """Registering a layer attaches a pre-forward hook to that layer. This function
        checks if the pre-forward hook works as expected. Specifically, checks if the
        input is aggregated correctly.

        Basically, asserts that the aggregate of input activations is the same as what was
        computed in the sparsifier.

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            data_list (list of torch tensors)
                data input to the model attached to the sparsifier

        """
        # can only check for the first layer
        data_agg_actual = data_list[0]
        model = activation_sparsifier.model
        layer_name = module_to_fqn(model, model.conv1)
        agg_fn = activation_sparsifier.data_groups[layer_name]["aggregate_fn"]

        for i in range(1, len(data_list)):
            data_agg_actual = agg_fn(data_agg_actual, data_list[i])

        if "data" not in activation_sparsifier.data_groups[layer_name]:
            raise AssertionError(f"Expected 'data' key in data_groups['{layer_name}']")
        if not torch.all(
            activation_sparsifier.data_groups[layer_name]["data"] == data_agg_actual
        ):
            raise AssertionError("Aggregated data does not match expected")

        return data_agg_actual

    def _check_step(self, activation_sparsifier, data_agg_actual):
        """Checks if .step() works as expected. Specifically, checks if the mask is computed correctly.

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            data_agg_actual (torch tensor)
                aggregated torch tensor

        """
        model = activation_sparsifier.model
        layer_name = module_to_fqn(model, model.conv1)
        if layer_name is None:
            raise AssertionError("Expected layer_name to be not None")

        reduce_fn = activation_sparsifier.data_groups[layer_name]["reduce_fn"]

        data_reduce_actual = reduce_fn(data_agg_actual)
        mask_fn = activation_sparsifier.data_groups[layer_name]["mask_fn"]
        sparse_config = activation_sparsifier.data_groups[layer_name]["sparse_config"]
        mask_actual = mask_fn(data_reduce_actual, **sparse_config)

        mask_model = activation_sparsifier.get_mask(layer_name)

        if not torch.all(mask_model == mask_actual):
            raise AssertionError("Mask from sparsifier does not match computed mask")

        for config in activation_sparsifier.data_groups.values():
            if "data" in config:
                raise AssertionError(
                    "Expected 'data' to be removed from config after step"
                )

    def _check_squash_mask(self, activation_sparsifier, data):
        """Makes sure that squash_mask() works as usual. Specifically, checks
        if the sparsifier hook is attached correctly.
        This is achieved by only looking at the identity layers and making sure that
        the output == layer(input * mask).

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            data (torch tensor)
                dummy batched data
        """

        # create a forward hook for checking output == layer(input * mask)
        def check_output(name):
            mask = activation_sparsifier.get_mask(name)
            features = activation_sparsifier.data_groups[name].get("features")
            feature_dim = activation_sparsifier.data_groups[name].get("feature_dim")

            def hook(module, input, output):
                input_data = input[0]
                if features is None:
                    if not torch.all(mask * input_data == output):
                        raise AssertionError("Output does not match mask * input")
                else:
                    for feature_idx in range(len(features)):
                        feature = torch.Tensor(
                            [features[feature_idx]], device=input_data.device
                        ).long()
                        inp_data_feature = torch.index_select(
                            input_data, feature_dim, feature
                        )
                        out_data_feature = torch.index_select(
                            output, feature_dim, feature
                        )

                        if not torch.all(
                            mask[feature_idx] * inp_data_feature == out_data_feature
                        ):
                            raise AssertionError(
                                f"Output for feature {feature_idx} does not match mask * input"
                            )

            return hook

        for name, config in activation_sparsifier.data_groups.items():
            if "identity" in name:
                config["layer"].register_forward_hook(check_output(name))

        activation_sparsifier.model(data)

    def _check_state_dict(self, sparsifier1):
        """Checks if loading and restoring of state_dict() works as expected.
        Basically, dumps the state of the sparsifier and loads it in the other sparsifier
        and checks if all the configuration are in line.

        This function is called at various times in the workflow to makes sure that the sparsifier
        can be dumped and restored at any point in time.
        """
        state_dict = sparsifier1.state_dict()

        new_model = Model()

        # create an empty new sparsifier
        sparsifier2 = ActivationSparsifier(new_model)

        if sparsifier2.defaults == sparsifier1.defaults:
            raise AssertionError(
                "Expected sparsifier defaults to be different before load"
            )
        if len(sparsifier2.data_groups) == len(sparsifier1.data_groups):
            raise AssertionError(
                "Expected data_groups lengths to be different before load"
            )

        sparsifier2.load_state_dict(state_dict)

        if sparsifier2.defaults != sparsifier1.defaults:
            raise AssertionError("Expected sparsifier defaults to match after load")

        for name, state in sparsifier2.state.items():
            if name not in sparsifier1.state:
                raise AssertionError(f"Expected '{name}' in sparsifier1.state")
            mask1 = sparsifier1.state[name]["mask"]
            mask2 = state["mask"]

            if mask1 is None:
                if mask2 is not None:
                    raise AssertionError("Expected mask2 to be None when mask1 is None")
            else:
                if type(mask1) is not type(mask2):
                    raise AssertionError(
                        f"Expected same mask types, got {type(mask1)} and {type(mask2)}"
                    )
                if isinstance(mask1, list):
                    if len(mask1) != len(mask2):
                        raise AssertionError(
                            f"Expected mask lengths to match, got {len(mask1)} and {len(mask2)}"
                        )
                    for idx in range(len(mask1)):
                        if not torch.all(mask1[idx] == mask2[idx]):
                            raise AssertionError(f"Masks at index {idx} do not match")
                else:
                    if not torch.all(mask1 == mask2):
                        raise AssertionError("Masks do not match")

        # make sure that the state dict is stored as torch sparse
        for state in state_dict["state"].values():
            mask = state["mask"]
            if mask is not None:
                if isinstance(mask, list):
                    for idx in range(len(mask)):
                        if not mask[idx].is_sparse:
                            raise AssertionError(f"Expected mask[{idx}] to be sparse")
                else:
                    if not mask.is_sparse:
                        raise AssertionError("Expected mask to be sparse")

        dg1, dg2 = sparsifier1.data_groups, sparsifier2.data_groups

        for layer_name, config in dg1.items():
            if layer_name not in dg2:
                raise AssertionError(f"Expected '{layer_name}' in dg2")

            # exclude hook and layer
            config1 = {
                key: value
                for key, value in config.items()
                if key not in ["hook", "layer"]
            }
            config2 = {
                key: value
                for key, value in dg2[layer_name].items()
                if key not in ["hook", "layer"]
            }

            if config1 != config2:
                raise AssertionError(f"Configs for '{layer_name}' do not match")

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_activation_sparsifier(self):
        """Simulates the workflow of the activation sparsifier, starting from object creation
        till squash_mask().
        The idea is to check that everything works as expected while in the workflow.
        """

        # defining aggregate, reduce and mask functions
        def agg_fn(x, y):
            return x + y

        def reduce_fn(x):
            return torch.mean(x, dim=0)

        def _vanilla_norm_sparsifier(data, sparsity_level):
            r"""Similar to data norm sparsifier but block_shape = (1,1).
            Simply, flatten the data, sort it and mask out the values less than threshold
            """
            data_norm = torch.abs(data).flatten()
            _, sorted_idx = torch.sort(data_norm)
            threshold_idx = round(sparsity_level * len(sorted_idx))
            sorted_idx = sorted_idx[:threshold_idx]

            mask = torch.ones_like(data_norm)
            mask.scatter_(dim=0, index=sorted_idx, value=0)
            mask = mask.reshape(data.shape)

            return mask

        # Creating default function and sparse configs
        # default sparse_config
        sparse_config = {"sparsity_level": 0.5}

        defaults = {"aggregate_fn": agg_fn, "reduce_fn": reduce_fn}

        # simulate the workflow
        # STEP 1: make data and activation sparsifier object
        model = Model()  # create model
        activation_sparsifier = ActivationSparsifier(model, **defaults, **sparse_config)

        # Test Constructor
        self._check_constructor(activation_sparsifier, model, defaults, sparse_config)

        # STEP 2: Register some layers
        register_layer1_args = {
            "layer": model.conv1,
            "mask_fn": _vanilla_norm_sparsifier,
        }
        sparse_config_layer1 = {"sparsity_level": 0.3}

        register_layer2_args = {
            "layer": model.linear1,
            "features": [0, 10, 234],
            "feature_dim": 1,
            "mask_fn": _vanilla_norm_sparsifier,
        }
        sparse_config_layer2 = {"sparsity_level": 0.1}

        register_layer3_args = {
            "layer": model.identity1,
            "mask_fn": _vanilla_norm_sparsifier,
        }
        sparse_config_layer3 = {"sparsity_level": 0.3}

        register_layer4_args = {
            "layer": model.identity2,
            "features": [0, 10, 20],
            "feature_dim": 1,
            "mask_fn": _vanilla_norm_sparsifier,
        }
        sparse_config_layer4 = {"sparsity_level": 0.1}

        layer_args_list = [
            (register_layer1_args, sparse_config_layer1),
            (register_layer2_args, sparse_config_layer2),
        ]
        layer_args_list += [
            (register_layer3_args, sparse_config_layer3),
            (register_layer4_args, sparse_config_layer4),
        ]

        # Registering..
        for layer_args in layer_args_list:
            layer_arg, sparse_config_layer = layer_args
            activation_sparsifier.register_layer(**layer_arg, **sparse_config_layer)

        # check if things are registered correctly
        self._check_register_layer(
            activation_sparsifier, defaults, sparse_config, layer_args_list
        )

        # check state_dict after registering and before model forward
        self._check_state_dict(activation_sparsifier)

        # check if forward pre hooks actually work
        # some dummy data
        data_list = []
        num_data_points = 5
        for _ in range(num_data_points):
            rand_data = torch.randn(16, 1, 28, 28)
            activation_sparsifier.model(rand_data)
            data_list.append(rand_data)

        data_agg_actual = self._check_pre_forward_hook(activation_sparsifier, data_list)
        # check state_dict() before step()
        self._check_state_dict(activation_sparsifier)

        # STEP 3: sparsifier step
        activation_sparsifier.step()

        # check state_dict() after step() and before squash_mask()
        self._check_state_dict(activation_sparsifier)

        # self.check_step()
        self._check_step(activation_sparsifier, data_agg_actual)

        # STEP 4: squash mask
        activation_sparsifier.squash_mask()

        self._check_squash_mask(activation_sparsifier, data_list[0])

        # check state_dict() after squash_mask()
        self._check_state_dict(activation_sparsifier)


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")
