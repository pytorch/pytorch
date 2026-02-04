# Owner(s): ["module: sparse"]

import copy
import itertools
import math

import torch
from torch import nn
from torch.ao.pruning._experimental.data_sparsifier import (
    BaseDataSparsifier,
    DataNormSparsifier,
)
from torch.ao.pruning._experimental.data_sparsifier.quantization_utils import (
    post_training_sparse_quantize,
)
from torch.nn.utils.parametrize import is_parametrized
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class ImplementedSparsifier(BaseDataSparsifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_mask(self, name, data, **kwargs):
        mask = self.get_mask(name)
        mask[0] = 0
        linear_state = self.state[name]
        linear_state["step_count"] = linear_state.get("step_count", 0) + 1


class _BaseDataSparsiferTestCase(TestCase):
    r"""This helper test class takes in any supported type of and runs some tests.
    The user is required to pass in the data that needs to sparsified and the
    runner will run some tests that needs to be passed in order for the data
    type to be supported.
    TODO: Change the structure by creating a separate test case class for each
          member function
    """

    def run_all_checks(self, data_list, data_with_config, defaults):
        self.check_constructor(data_list, data_with_config, defaults)
        self.check_squash_mask(data_list, data_with_config, defaults)
        self.check_add_data(data_list, data_with_config, defaults)
        self.check_step(data_list, data_with_config, defaults)
        self.check_state_dict(data_list, data_with_config, defaults)
        self.check_memory_reference(data_list, data_with_config, defaults)

    @staticmethod
    def _get_name_data_config(some_data, defaults=None):
        if isinstance(some_data, tuple):
            # dealing with data_list
            name, data = some_data
            config = defaults
        else:
            # dealing with data_with_config
            name, data, config = (
                some_data["name"],
                some_data["data"],
                some_data["config"],
            )
        return name, data, config

    @staticmethod
    def _make_sparsifier(
        data_list,
        data_with_config,
        defaults,
        sparsifier_type=None,
        sparsifier_kwargs=None,
    ):
        if sparsifier_type is None:
            sparsifier = ImplementedSparsifier(data_list=data_list, **defaults)
        else:
            kwargs = copy.deepcopy(defaults)
            kwargs.update(sparsifier_kwargs)
            kwargs["data_list"] = data_list
            sparsifier = sparsifier_type(**kwargs)
        if len(sparsifier.data_groups) != len(data_list):
            raise AssertionError(
                f"Expected {len(data_list)} data_groups, got {len(sparsifier.data_groups)}"
            )
        for data_config_dict in data_with_config:
            name, data, config = (
                data_config_dict["name"],
                data_config_dict["data"],
                data_config_dict["config"],
            )
            sparsifier.add_data(name=name, data=data, **config)
        return sparsifier

    def check_constructor(self, data_list, data_with_config, defaults, **kwargs):
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        self.assertEqual(
            len(sparsifier.data_groups),
            len(data_list) + len(data_with_config),
            msg="Sparsifier data groups don't match the input "
            f"({len(sparsifier.data_groups)} vs. "
            f"{len(data_list) + len(data_with_config)}).",
        )

        all_data = data_list + data_with_config

        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data, defaults=defaults)
            self.assertIn(name, sparsifier.data_groups)
            self.assertEqual(sparsifier.data_groups[name], config)

    def check_step(self, data_list, data_with_config, defaults, **kwargs):
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        all_data = data_list + data_with_config

        # Check data and mask before doing the step
        for some_data in all_data:
            name, data, _ = self._get_name_data_config(some_data)
            data = sparsifier._extract_weight(data)
            sparsified_data = sparsifier.get_data(name=name, return_original=False)
            original_data = sparsifier.get_data(name=name, return_original=True)
            mask = sparsifier.get_mask(name=name)
            self.assertEqual(sparsified_data, data)
            self.assertEqual(original_data, data)
            self.assertEqualBroadcasting(mask[0], 1)

        step_count = 3

        for _ in range(step_count):
            sparsifier.step()
        for some_data in all_data:
            name, data, _ = self._get_name_data_config(some_data)
            data = sparsifier._extract_weight(data)
            sparsified_data = sparsifier.get_data(name=name, return_original=False)
            original_data = sparsifier.get_data(name=name, return_original=True)
            mask = sparsifier.get_mask(name=name)
            self.assertEqualBroadcasting(sparsified_data[0], 0)
            self.assertEqual(original_data, data)
            self.assertEqualBroadcasting(mask[0], 0)
            if "step_count" not in sparsifier.state[name]:
                raise AssertionError(
                    f"Expected 'step_count' in sparsifier.state['{name}']"
                )
            if sparsifier.state[name]["step_count"] != 3:
                raise AssertionError(
                    f"Expected step_count 3, got {sparsifier.state[name]['step_count']}"
                )

    def check_squash_mask(self, data_list, data_with_config, defaults, **kwargs):
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        all_data = data_list + data_with_config
        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            if not hasattr(sparsifier._container, name):
                raise AssertionError(f"Expected container to have attribute '{name}'")
            if not is_parametrized(sparsifier._container, name):
                raise AssertionError(f"Expected container.{name} to be parametrized")
        sparsifier.step()
        sparsifier.squash_mask()

        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            if is_parametrized(sparsifier._container, name):
                raise AssertionError(
                    f"Expected container.{name} to not be parametrized after squash"
                )
            with self.assertRaises(ValueError):
                sparsifier.get_data(name, return_original=True)

    def check_add_data(self, data_list, data_with_config, defaults, **kwargs):
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        all_data = data_list + data_with_config
        for some_data in all_data:
            name1, data1, config = self._get_name_data_config(
                some_data, defaults=defaults
            )
            data1 = sparsifier._extract_weight(data1)
            data1_old = copy.deepcopy(data1)
            if not torch.all(data1 == sparsifier.get_data(name=name1)):
                raise AssertionError("data1 does not match sparsifier data")

            sparsifier.step()
            mask = sparsifier.get_mask(name1)

            data2 = torch.randn(
                data1.shape
            )  # add another data with the same shape as original data
            sparsifier.add_data(name=name1, data=data2)
            if not torch.all(data2 == sparsifier.get_data(name=name1)):
                raise AssertionError("data2 does not match sparsifier data")

            if not torch.all(sparsifier.get_mask(name1) == mask):
                raise AssertionError("mask should not change after add_data")
            if not torch.all(data1_old == data1):
                raise AssertionError("data1 should not be modified")

            if sparsifier.data_groups[name1] != config:
                raise AssertionError(
                    "old_config should match new config after replacement"
                )

    def check_state_dict(self, data_list, data_with_config, defaults, **kwargs):
        sparsifier1 = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        sparsifier2 = self._make_sparsifier(
            data_list=[data_list[0]], data_with_config=[], defaults=defaults, **kwargs
        )
        sparsifier1.step()

        state_dict1 = sparsifier1.state_dict()

        if sparsifier1.state == sparsifier2.state:
            raise AssertionError("Expected sparsifier states to be different")
        name, _, _ = self._get_name_data_config(data_list[0])
        self.assertNotEqual(sparsifier1.get_mask(name), sparsifier2.get_mask(name))

        sparsifier2.load_state_dict(state_dict1)
        if len(sparsifier1.state) != len(sparsifier2.state):
            raise AssertionError(
                f"Expected state lengths to match, got {len(sparsifier1.state)} vs {len(sparsifier2.state)}"
            )
        if len(sparsifier1.data_groups) != len(sparsifier2.data_groups):
            raise AssertionError(
                f"Expected data_groups lengths to match, got {len(sparsifier1.data_groups)} vs {len(sparsifier2.data_groups)}"
            )

        state1 = state_dict1["state"]
        for name in state1:
            # compare mask
            if name not in sparsifier2.state:
                raise AssertionError(f"Expected '{name}' in sparsifier2.state")
            if "mask" not in sparsifier2.state[name]:
                raise AssertionError(f"Expected 'mask' in sparsifier2.state['{name}']")
            if "mask" not in sparsifier1.state[name]:
                raise AssertionError(f"Expected 'mask' in sparsifier1.state['{name}']")
            mask1, mask2 = state1[name]["mask"], sparsifier2.state[name]["mask"]
            if not (mask1.is_sparse and not mask2.is_sparse):
                raise AssertionError(
                    "Expected mask1 to be sparse and mask2 to be dense"
                )
            if not torch.all(mask1.to_dense() == mask2):
                raise AssertionError("Masks do not match after loading state dict")

            # compare data_groups
            dg1, dg2 = sparsifier1.data_groups, sparsifier2.data_groups
            if not (name in dg1 and name in dg2):
                raise AssertionError(f"Expected '{name}' in both data_groups")
            if dg1[name] != dg2[name]:
                raise AssertionError(f"data_groups['{name}'] do not match")

            # compare container
            container1, container2 = sparsifier1._container, sparsifier2._container
            if not torch.all(getattr(container1, name) == getattr(container2, name)):
                raise AssertionError(f"Container data for '{name}' do not match")
            if is_parametrized(container1, name) != is_parametrized(container2, name):
                raise AssertionError(f"Parametrization state for '{name}' do not match")
            if is_parametrized(container1, name):
                param1 = getattr(container1.parametrizations, name)[0]
                param2 = getattr(container2.parametrizations, name)[0]
                if not hasattr(param1, "mask"):
                    raise AssertionError("Expected param1 to have 'mask' attribute")
                if not hasattr(param2, "mask"):
                    raise AssertionError("Expected param2 to have 'mask' attribute")
                self.assertEqual(param1.__dict__, param2.__dict__)

    def check_memory_reference(self, data_list, data_with_config, defaults, **kwargs):
        """Checks if the data is truly "attached" to the sparsifier. Meaning, when the
        data is changed outside of the sparsifier, the changes must be reflected on the data
        inside the data sparsifier as well.
        This makes sure that the sparsifier is holding the memory reference of the data and
        not copies.

        This test modifies the data and asserts that data in the sparsifier is changed as well
        """
        sparsifier = self._make_sparsifier(
            data_list, data_with_config, defaults=defaults, **kwargs
        )
        all_data = data_list + data_with_config
        for some_data in all_data:
            name, data, _ = self._get_name_data_config(some_data)
            weight = sparsifier._extract_weight(data)
            weight.data = weight + torch.randn(*weight.shape)
            contained_data = sparsifier.get_data(name=name)
            if (
                weight.data.storage().data_ptr()
                != contained_data.data.storage().data_ptr()
            ):
                raise AssertionError("Memory reference is not preserved")
            if not torch.all(contained_data == weight):
                raise AssertionError("Contained data does not match weight")


class _NormDataSparsifierTestCase(_BaseDataSparsiferTestCase):
    r"""This helper test class takes in any supported type of and runs some tests.
    This inherits the TestBaseDataSparsifierRuner wherein some functions are
    over-ridden to take accommodate the specific sparsifier.
    TODO: Change the structure by creating a separate test case class for each
          member function
    """

    def run_all_checks(self, data_list, defaults, data_with_config, norm_type="L1"):
        if norm_type not in ["L1", "L2"]:
            raise AssertionError(f"norm_type must be 'L1' or 'L2', got '{norm_type}'")
        kwargs = {
            "sparsifier_type": DataNormSparsifier,
            "sparsifier_kwargs": {"norm": norm_type},
        }
        self.check_constructor(data_list, data_with_config, defaults, **kwargs)
        self.check_squash_mask(data_list, data_with_config, defaults, **kwargs)
        self.check_add_data(data_list, data_with_config, defaults, **kwargs)
        self.check_state_dict(data_list, data_with_config, defaults, **kwargs)
        self.check_step(data_list, data_with_config, defaults, norm_type=norm_type)
        self.check_step_2_of_4(norm_type=norm_type)
        self.check_sparsity_level(
            data_list, data_with_config, defaults, norm_type=norm_type
        )
        self.check_memory_reference(data_list, data_with_config, defaults, **kwargs)

    @staticmethod
    def _get_bounds_on_actual_sparsity(config, tensor_shape):
        r"""This function gets the bounds on actual sparsity.
        Note::
            Although we specify the sparsity_level parameter, this does not mean that
            the actual sparsity obtained after sparsification is the same as sparsity_level.
            The actual sparsity depends largely on the shape and the data itself.
        """
        sparsity_level = config["sparsity_level"]
        zeros_per_block = config["zeros_per_block"]
        sparse_block_shape = config["sparse_block_shape"]

        height, width = tensor_shape[-2], tensor_shape[-1]
        block_height, block_width = sparse_block_shape
        number_blocks = math.ceil(height / block_height) * math.ceil(
            width / block_width
        )
        values_per_block = block_height * block_width

        if zeros_per_block == 0:
            return (1.0, 1.0)
        else:
            # min value assumes zeros_per_block is 1
            min_values_sparsified = round(number_blocks * sparsity_level)
            # max value assumes actual zeros_per_block
            max_values_sparsified = min_values_sparsified * min(
                values_per_block, zeros_per_block
            )
            lower_bound = min_values_sparsified / (height * width)
            upper_bound = min(1.0, max_values_sparsified / (height * width))

            lower_bound, upper_bound = round(lower_bound, 3), round(upper_bound, 3)
            return lower_bound, upper_bound

    def check_step(self, data_list, data_with_config, defaults, norm_type="L1"):
        sparsifier = self._make_sparsifier(
            data_list,
            data_with_config,
            defaults,
            sparsifier_type=DataNormSparsifier,
            sparsifier_kwargs={"norm": norm_type},
        )
        all_data = data_list + data_with_config

        # mask before step() should not be sparsified
        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            mask = sparsifier.get_mask(name=name)
            if (1.0 - mask.mean()) != 0:
                raise AssertionError("Expected sparsity level to be 0 before step")

        sparsifier.step()

        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            mask = sparsifier.get_mask(name=name)
            config = sparsifier.data_groups[name]
            lb, ub = self._get_bounds_on_actual_sparsity(config, mask.shape)
            mask = mask.to(torch.float)
            actual_sparsity = round(1 - mask.mean().item(), 3)
            if not (actual_sparsity >= lb and actual_sparsity <= ub):
                raise AssertionError(
                    f"Actual sparsity {actual_sparsity} not in bounds [{lb}, {ub}]"
                )
            if actual_sparsity <= 0.0:
                raise AssertionError("Actual sparsity should be > 0.0")

        iters_before_collapse = 100

        test_sparsifier = DataNormSparsifier(
            sparsity_level=0.5,
            sparse_block_shape=(1, 4),
            zeros_per_block=4,
            norm=norm_type,
        )

        for _ in range(iters_before_collapse):
            new_data = torch.randn(20, 20)
            test_sparsifier.add_data(name="test_data", data=new_data)
            test_sparsifier.step()
            mask = test_sparsifier.get_mask(name="test_data")
            mask = mask.to(torch.float)
            if (1.0 - mask.mean().item()) <= 0:
                raise AssertionError("Expected some sparsity to be achieved")

    def check_step_2_of_4(self, norm_type):
        # overriding default config for test purposes
        default_config = {
            "sparsity_level": 1.0,
            "zeros_per_block": 2,
            "sparse_block_shape": (1, 4),
        }
        data_list = [("test_data", torch.randn(4, 4))]

        sparsifier = DataNormSparsifier(
            data_list=data_list, norm=norm_type, **default_config
        )
        sparsifier.step()

        for some_data in data_list:
            name, _ = some_data
            mask = sparsifier.get_mask(name=name)
            mask = mask.to(torch.float)
            self.assertAlmostEqual(1.0 - mask.mean().item(), 0.5, places=2)
            for row in mask:
                for idx in range(0, len(row), 4):
                    block = row[idx : idx + 4]
                    block, _ = block.sort()
                    if not (block[:2] == 0).all():
                        raise AssertionError(
                            "Expected first 2 elements of block to be 0"
                        )
                    if not (block[2:] != 0).all():
                        raise AssertionError(
                            "Expected last 2 elements of block to be non-zero"
                        )

    def check_sparsity_level(
        self, data_list, data_with_config, defaults, norm_type="L1"
    ):
        sparsity_levels = [-1.0, 0.0, 0.5, 1.0, 2.0]
        sparse_block_shapes = [(1, 1), (1, 4), (2, 2), (4, 1)]
        zeros_per_blocks = [0, 1, 2, 3, 4]
        sparsifier = DataNormSparsifier(data_list=data_list, norm=norm_type)

        testcases = itertools.tee(
            itertools.product(sparsity_levels, sparse_block_shapes, zeros_per_blocks)
        )

        if not (
            len(data_with_config) > 0
            and "name" in data_with_config[0]
            and "data" in data_with_config[0]
        ):
            raise AssertionError(
                "data_with_config must have at least one entry with 'name' and 'data'"
            )
        # get some data
        name, data = data_with_config[0]["name"], data_with_config[0]["data"]
        for idx, (sl, sbs, zpb) in enumerate(testcases[0]):
            new_name = f"{name}_{idx}"
            if zpb > sbs[0] * sbs[1]:
                continue
            current_config = {
                "sparsity_level": sl,
                "sparse_block_shape": sbs,
                "zeros_per_block": zpb,
            }
            sparsifier.add_data(name=new_name, data=data, **current_config)
            if zpb > sbs[0] * sbs[1]:
                continue

        sparsifier.step()
        sparsifier.squash_mask()
        for idx, (sl, sbs, zpb) in enumerate(testcases[0]):
            new_name = f"{name}_{idx}"
            sparsified_data = sparsifier.get_data(name=new_name, original=False)
            # sparse mask
            sparse_mask = (sparsified_data == 0).float()
            if zpb == 0:
                if sparse_mask.mean() != 0:
                    raise AssertionError(
                        f"Expected sparse_mask.mean() == 0, got {sparse_mask.mean()}"
                    )
            else:
                # Ratio of individual zeros in the tensor
                true_sl = min(max(sl, 0.0), 1.0)
                true_sl = true_sl * zpb / sbs[0] / sbs[1]
                if sparse_mask.mean() != true_sl:
                    raise AssertionError(
                        f"Expected sparse_mask.mean() == {true_sl}, got {sparse_mask.mean()}"
                    )


class TestBaseDataSparsifier(_BaseDataSparsiferTestCase):
    """To add unit tests to support new data types for the BaseDataSparsifier, create the following
        data_list: List of tuples of name, data to be added to the constructor
        defaults: default config for the above data in data_list
        data_with_config: list of dictionaries defining name, data and config (look test_tensors())

    Once the above is done, create an instance of TestBaseDataSparsifierType and call all the run_tests()
    """

    def test_tensors(self):
        tensor1, tensor2, tensor3 = (
            torch.randn(3, 3),
            torch.randn(4, 4),
            torch.randn(5, 5),
        )
        tensor4, tensor5 = torch.randn(1, 1), torch.randn(4, 4)
        data_list = [("tensor1", tensor1), ("tensor2", tensor2), ("tensor3", tensor3)]
        defaults = {"test": 3}

        data_with_config = [
            {"name": "tensor4", "data": tensor4, "config": {"test": 7}},
            {"name": "tensor5", "data": tensor5, "config": {"test": 8}},
        ]
        self.run_all_checks(
            data_list=data_list, defaults=defaults, data_with_config=data_with_config
        )

    def test_nn_parameters(self):
        param1, param2, param3 = (
            nn.Parameter(torch.randn(3, 3)),
            nn.Parameter(torch.randn(4, 4)),
            nn.Parameter(torch.randn(5, 5)),
        )
        param4, param5 = (
            nn.Parameter(torch.randn(1, 1)),
            nn.Parameter(torch.randn(4, 4)),
        )
        data_list = [("param1", param1), ("param2", param2), ("param3", param3)]
        defaults = {"test": 3}

        data_with_config = [
            {"name": "param4", "data": param4, "config": {"test": 7}},
            {"name": "param5", "data": param5, "config": {"test": 8}},
        ]
        self.run_all_checks(
            data_list=data_list, defaults=defaults, data_with_config=data_with_config
        )

    def test_nn_embeddings(self):
        (
            emb1,
            emb2,
        ) = nn.Embedding(10, 3), nn.Embedding(20, 3)
        emb1_bag, emb2_bag = nn.EmbeddingBag(10, 3), nn.EmbeddingBag(20, 3)

        emb3, emb3_bag = nn.Embedding(15, 3), nn.EmbeddingBag(20, 3)
        data_list = [
            ("emb1", emb1),
            ("emb1_bag", emb1_bag),
            ("emb2", emb2),
            ("emb2_bag", emb2_bag),
        ]
        defaults = {"test": 3}

        data_with_config = [
            {"name": "emb3", "data": emb3, "config": {"test": 7}},
            {"name": "emb3_bag", "data": emb3_bag, "config": {"test": 8}},
        ]
        self.run_all_checks(
            data_list=data_list, defaults=defaults, data_with_config=data_with_config
        )


class TestNormDataSparsifiers(_NormDataSparsifierTestCase):
    """To add unit tests to support new data types for the NormDataSparsifier, create the following
    data_list: List of tuples of name, data to be added to the constructor
    defaults: default config for the above data in data_list
    data_with_config: list of dictionaries defining name, data and config (look test_tensors())

    Once the above is done, create an instance of _NormDataSparsifierTestRunner and call run_tests()
    """

    def test_tensors(self):
        tensor1, tensor2, tensor3 = (
            torch.randn(1, 10),
            torch.randn(4, 4),
            torch.randn(1, 5),
        )
        tensor4, tensor5 = torch.randn(1, 2), torch.randn(4, 4)
        data_list = [("tensor1", tensor1), ("tensor2", tensor2), ("tensor3", tensor3)]
        defaults = {
            "sparsity_level": 0.5,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 4,
        }

        data_with_config = [
            {
                "name": "tensor4",
                "data": tensor4,
                "config": {
                    "sparsity_level": 0.7,
                    "sparse_block_shape": (2, 3),
                    "zeros_per_block": 6,
                },
            },
            {
                "name": "tensor5",
                "data": tensor5,
                "config": {
                    "sparsity_level": 0.3,
                    "sparse_block_shape": (2, 3),
                    "zeros_per_block": 6,
                },
            },
        ]
        self.run_all_checks(
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L1",
        )
        self.run_all_checks(
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L2",
        )

    def test_nn_parameters(self):
        param1, param2, param3 = (
            nn.Parameter(torch.randn(1, 8)),
            nn.Parameter(torch.randn(4, 4)),
            nn.Parameter(torch.randn(5, 5)),
        )
        param4, param5 = (
            nn.Parameter(torch.randn(10, 10)),
            nn.Parameter(torch.randn(4, 4)),
        )
        data_list = [("param1", param1), ("param2", param2), ("param3", param3)]
        defaults = {
            "sparsity_level": 0.5,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 4,
        }

        data_with_config = [
            {
                "name": "param4",
                "data": param4,
                "config": {
                    "sparsity_level": 0.7,
                    "sparse_block_shape": (2, 3),
                    "zeros_per_block": 6,
                },
            },
            {
                "name": "param5",
                "data": param5,
                "config": {
                    "sparsity_level": 0.3,
                    "sparse_block_shape": (2, 3),
                    "zeros_per_block": 6,
                },
            },
        ]
        self.run_all_checks(
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L1",
        )
        self.run_all_checks(
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L2",
        )

    def test_nn_embeddings(self):
        (
            emb1,
            emb2,
        ) = nn.Embedding(10, 3), nn.Embedding(20, 3)
        emb1_bag, emb2_bag = nn.EmbeddingBag(10, 3), nn.EmbeddingBag(20, 3)

        emb3, emb3_bag = nn.Embedding(15, 3), nn.EmbeddingBag(20, 3)
        data_list = [
            ("emb1", emb1),
            ("emb1_bag", emb1_bag),
            ("emb2", emb2),
            ("emb2_bag", emb2_bag),
        ]
        defaults = {
            "sparsity_level": 0.5,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 4,
        }

        data_with_config = [
            {
                "name": "emb3",
                "data": emb3,
                "config": {
                    "sparsity_level": 0.7,
                    "sparse_block_shape": (2, 3),
                    "zeros_per_block": 6,
                },
            },
            {
                "name": "emb3_bag",
                "data": emb3_bag,
                "config": {
                    "sparsity_level": 0.3,
                    "sparse_block_shape": (2, 3),
                    "zeros_per_block": 6,
                },
            },
        ]
        self.run_all_checks(
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L1",
        )

        self.run_all_checks(
            data_list=data_list,
            defaults=defaults,
            data_with_config=data_with_config,
            norm_type="L2",
        )


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb1 = nn.Embedding(100, 3)
        self.embbag1 = nn.EmbeddingBag(200, 32)
        self.emb_seq = nn.Sequential(nn.Embedding(150, 3), nn.EmbeddingBag(100, 3))
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(16, 16)


class TestQuantizationUtils(TestCase):
    def test_ptq_sparsify_first(self):
        """The expectation is post_training_sparse_quantize function
        1. Takes in a model
        2. Sparsifies the embeddings
        3. Quantize the embeddings

        This unit test checks that
        1. Embeddings and EmbeddingBags are sparsified to the right sparsity levels
        2. Embeddings and EmbeddingBags are quantized
        3. Linear modules are not quantized
        """
        model = Model()

        sparse_config = {"sparsity_level": 0.80, "sparse_block_shape": (1, 1)}
        select_embeddings = [model.embbag1, model.emb1]
        post_training_sparse_quantize(
            model,
            data_sparsifier_class=DataNormSparsifier,
            sparsify_first=True,
            select_embeddings=select_embeddings,
            **sparse_config,
        )

        if (
            type(model.emb1)
            is not torch.ao.nn.quantized.modules.embedding_ops.Embedding
        ):
            raise AssertionError(
                f"Expected quantized Embedding, got {type(model.emb1)}"
            )
        if (
            type(model.embbag1)
            is not torch.ao.nn.quantized.modules.embedding_ops.EmbeddingBag
        ):
            raise AssertionError(
                f"Expected quantized EmbeddingBag, got {type(model.embbag1)}"
            )
        if type(model.emb_seq[0]) is not nn.Embedding:
            raise AssertionError(f"Expected nn.Embedding, got {type(model.emb_seq[0])}")
        if type(model.emb_seq[1]) is not nn.EmbeddingBag:
            raise AssertionError(
                f"Expected nn.EmbeddingBag, got {type(model.emb_seq[1])}"
            )
        if type(model.linear1) is not nn.Linear:
            raise AssertionError(f"Expected nn.Linear, got {type(model.linear1)}")
        if type(model.linear2) is not nn.Linear:
            raise AssertionError(f"Expected nn.Linear, got {type(model.linear2)}")

        dequant_emb1 = torch.dequantize(model.emb1.weight())
        dequant_embbag1 = torch.dequantize(model.embbag1.weight())

        threshold = 1e-2

        sl_emb1 = (torch.abs(dequant_emb1) < threshold).float().mean()
        sl_embbag1 = (torch.abs(dequant_embbag1) < threshold).float().mean()

        if abs(sl_emb1 - 0.80) > 0.05:
            raise AssertionError(f"Expected sl_emb1 ~0.80, got {sl_emb1}")
        if abs(sl_embbag1 - 0.80) > 0.05:
            raise AssertionError(f"Expected sl_embbag1 ~0.80, got {sl_embbag1}")

    def test_ptq_quantize_first(self):
        """The expectation is post_training_sparse_quantize function
        1. Takes in a model
        2. Quantize the embeddings
        3. Sparsifies the embeddings

        This unit test checks that
        1. Embeddings and EmbeddingBags are sparsified to the right sparsity levels
        2. Embeddings and EmbeddingBags are quantized
        3. Linear modules are not quantized
        """
        model = Model()

        sparse_config = {"sparsity_level": 0.8, "sparse_block_shape": (1, 1)}
        post_training_sparse_quantize(
            model, DataNormSparsifier, sparsify_first=False, **sparse_config
        )

        if (
            type(model.emb1)
            is not torch.ao.nn.quantized.modules.embedding_ops.Embedding
        ):
            raise AssertionError(
                f"Expected quantized Embedding, got {type(model.emb1)}"
            )
        if (
            type(model.embbag1)
            is not torch.ao.nn.quantized.modules.embedding_ops.EmbeddingBag
        ):
            raise AssertionError(
                f"Expected quantized EmbeddingBag, got {type(model.embbag1)}"
            )
        if (
            type(model.emb_seq[0])
            is not torch.ao.nn.quantized.modules.embedding_ops.Embedding
        ):
            raise AssertionError(
                f"Expected quantized Embedding, got {type(model.emb_seq[0])}"
            )
        if (
            type(model.emb_seq[1])
            is not torch.ao.nn.quantized.modules.embedding_ops.EmbeddingBag
        ):
            raise AssertionError(
                f"Expected quantized EmbeddingBag, got {type(model.emb_seq[1])}"
            )
        if type(model.linear1) is not nn.Linear:
            raise AssertionError(
                f"Expected nn.Linear (not quantized), got {type(model.linear1)}"
            )
        if type(model.linear2) is not nn.Linear:
            raise AssertionError(
                f"Expected nn.Linear (not quantized), got {type(model.linear2)}"
            )

        dequant_emb1 = torch.dequantize(model.emb1.weight())
        dequant_embbag1 = torch.dequantize(model.embbag1.weight())
        dequant_emb_seq_0 = torch.dequantize(model.emb_seq[0].weight())
        dequant_emb_seq_1 = torch.dequantize(model.emb_seq[1].weight())

        # higher threshold as quantization occurs before sparsity
        threshold = (
            1  # zero points seem to have higher magnitude with sparsity occurring after
        )

        sl_emb1 = (torch.abs(dequant_emb1) < threshold).float().mean()
        sl_embbag1 = (torch.abs(dequant_embbag1) < threshold).float().mean()
        sl_emb_seq_0 = (torch.abs(dequant_emb_seq_0) < threshold).float().mean()
        sl_emb_seq_1 = (torch.abs(dequant_emb_seq_1) < threshold).float().mean()

        if abs(sl_emb1 - 0.80) > 0.05:
            raise AssertionError(f"Expected sl_emb1 ~0.80, got {sl_emb1}")
        if abs(sl_embbag1 - 0.80) > 0.05:
            raise AssertionError(f"Expected sl_embbag1 ~0.80, got {sl_embbag1}")
        if abs(sl_emb_seq_0 - 0.80) > 0.05:
            raise AssertionError(f"Expected sl_emb_seq_0 ~0.80, got {sl_emb_seq_0}")
        if abs(sl_emb_seq_1 - 0.80) > 0.05:
            raise AssertionError(f"Expected sl_emb_seq_1 ~0.80, got {sl_emb_seq_1}")


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")
