import copy
import dataclasses
import functools
import itertools
import logging
import os
import pickle
import re
from enum import IntEnum
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import tqdm

import torch
import torch._logging
from torch import nn
from torch._inductor import config, dependencies, scheduler
from torch._inductor.coordinate_descent_tuner import CoordescTuner, get_field, set_field
from torch._inductor.dependencies import StarDep, WeakDep
from torch._inductor.triton_heuristics import (
    persistent_reduction_heuristic,
    pointwise_heuristic,
    reduction_heuristic,
    unique_configs,
)
from torch._inductor.utils import get_dtype_size, sympy_product
from torch._inductor.virtualized import V

log = logging.getLogger(__name__)


# Raw data related
class AutotunerRawData(NamedTuple):
    """AutotunerRawData is the raw data containing all the necessary information of the kernel.

    - When training the model
        we dump the raw data to a pickle file and load it to do feature extraction.
        this allows us to try different feature vectors without recompiling the kernels.
    - When predicting the best configs in e2e run
        we first extract the raw data from the kernel and then call the feature extraction
        function of the model we are using.
        see autotuner_predict() method below
    """

    io_deps: Tuple[
        List[dependencies.Dep], List[dependencies.Dep], List[int], List[int], List[int]
    ]
    read_write: dependencies.ReadWrites
    src_code: str
    autotuner_dict: Dict

    def __repr__(self):
        return f"AutotunerRawData({self.io_deps}, {self.read_write}, {self.src_code}, {self.autotuner_dict})"


def get_reads_writes(cur_scheduler, node, src_code):
    """Get the read and write info of the kernel, note that it's dependency-wise,
    which means if some kernel loads a tensor multiple times, there will be multiple
    read deps for that tensor.
    """
    if isinstance(node, scheduler.NopKernelSchedulerNode):
        return 0
    reads = {dep.name for dep in node.read_writes.reads}
    writes = {dep.name for dep in node.read_writes.writes}

    def is_materialized(buf):
        buf_uses = {user.node for user in cur_scheduler.name_to_node[buf].users}
        return len(buf_uses - set(node.snodes)) > 0

    if isinstance(node, scheduler.FusedSchedulerNode):
        removed_buffers = {dep for dep in writes if not is_materialized(dep)}
        writes = writes - removed_buffers
        reads = reads - removed_buffers

    dep_removed = list()
    strides = list()
    sizes = list()
    node_bytes = list()

    # To have deterministic order
    def f(dep_set):
        for dep in sorted(dep_set, key=lambda x: x.name):
            if dep.name not in reads | writes:
                continue
            if dep.name in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[dep.name]
            elif dep.name in V.graph.graph_inputs:
                buf = V.graph.graph_inputs[dep.name]
            else:
                continue

            dep_removed.append(dep)
            if isinstance(dep, (StarDep, WeakDep)):
                strides.append(None)
                sizes.append(None)
                node_bytes.append(
                    V.graph.sizevars.size_hint(sympy_product(buf.get_size()))
                    * get_dtype_size(buf.get_dtype())
                )
            else:
                strides.append(V.graph.sizevars.stride_hints(dep.index, dep.var_names))
                sizes.append([V.graph.sizevars.size_hint(s) for s in dep.size])
                node_bytes.append(
                    V.graph.sizevars.size_hint(sympy_product(dep.size))
                    * get_dtype_size(buf.get_dtype())
                )

    f(node.read_writes.reads)
    read_len = len(dep_removed)
    f(node.read_writes.writes)
    assert len(dep_removed) == len(node_bytes) == len(strides) == len(sizes)
    return (
        dep_removed[:read_len],
        dep_removed[read_len:],
        strides,
        sizes,
        node_bytes,
    )


# feature extraction related

# op_dict needs to be deterministic
# mapping Triton ops to integers
OP_DICT = {
    "load": 0,
    "to_dtype": 1,
    "add": 2,
    "reduction": 3,
    "constant": 4,
    "div": 5,
    "store": 6,
    "sub": 7,
    "square": 8,
    "rsqrt": 9,
    "mul": 10,
    "tanh": 11,
    "ne": 12,
    "where": 13,
    "indirect_indexing": 14,
    "log": 15,
    "neg": 16,
    "exp": 17,
    "maximum": 18,
    "minimum": 19,
    "index_expr": 20,
    "ge": 21,
    "masked": 22,
    "lt": 23,
    "and_": 24,
    "erf": 25,
    "eq": 26,
    "le": 27,
    "gt": 28,
    "relu": 29,
    "sqrt": 30,
    "logical_not": 31,
    "load_seed": 32,
    "rand": 33,
    "abs": 34,
    "reciprocal": 35,
    "ceil": 36,
    "sigmoid": 37,
    "sin": 38,
    "cos": 39,
    "logical_and": 40,
    "bitwise_and": 41,
    "randn": 42,
    "floor": 43,
    "remainder": 44,
    "isinf": 45,
    "logical_or": 46,
    "expm1": 47,
    "libdevice_sqrt": 48,
    "libdevice_log": 49,
    "truediv": 50,
    "sign": 51,
    "randint64": 52,
    "bitwise_or": 53,
    "pow": 54,
    "isnan": 55,
    "store_reduction": 56,
}


@dataclasses.dataclass
class DepFeatureVector:
    """Features related to a single dependency"""

    StarDepOrWeakDep: bool
    bytes: int
    strides: np.array
    sizes: np.array
    is_contiguous: bool
    is_scalar: bool
    is_indirect: bool


@dataclasses.dataclass
class KernelFeatureVector:
    """Features related to a single kernel"""

    kernel_category: int
    num_of_loops: int
    op_vec: np.array
    size_hints: np.array
    read_deps: List[DepFeatureVector]
    write_deps: List[DepFeatureVector]
    numels: np.array


@dataclasses.dataclass
class FeatureVector:
    kernel_feature: KernelFeatureVector
    XBLOCK: int
    YBLOCK: int
    RBLOCK: int
    num_warps: int
    num_stages: int


class KernelCategory(IntEnum):
    """Currently there are 3 kinds of kernels:
    1. pointwise kernels
    2. reduction kernels
    3. persistent reduction kernels
    """

    POINTWISE = 0
    REDUCTION = 1
    PERSISTENT_REDUCTION = 2


def get_kernel_category(src: str) -> KernelCategory:
    if "@pointwise" in src:
        return KernelCategory.POINTWISE
    if "@reduction" in src:
        return KernelCategory.REDUCTION
    if "@persistent_reduction" in src:
        return KernelCategory.PERSISTENT_REDUCTION
    raise AssertionError("Unknown kernel category")


def get_number_of_loops(src: str) -> int:
    return src.count("for roffset in range(0, rnumel, RBLOCK):")


def parse_list_of_numbers(s: str) -> list:
    # num1, num2, num3, ...
    nums = s.strip().split(",")
    nums = [num.strip() for num in nums]
    return [int(num) for num in nums]


def get_size_hints(src: str) -> list:
    return parse_list_of_numbers(re.search(r"size_hints=\[([^\]]*)\]", src).group(1))


def get_tiling(src: str) -> list:
    names = ["xnumel", "ynumel", "rnumel"]
    result = list()
    for name in names:
        startpos = src.find(name + " =")
        if startpos == -1:
            result.append(1)
            continue
        endpos = src.find("\n", startpos)
        result.append(int(src[startpos + len(name + " = ") : endpos]))
    return result


def make_dep_feature(
    ndims_lim,
    StarDepOrWeakDep: bool = False,
    bytes: int = 0,
    strides: List[int] = None,
    sizes: List[int] = None,
    is_contiguous: bool = True,
    is_scalar: bool = False,
    is_indirect: bool = False,
):
    tensor_feature = DepFeatureVector(None, None, None, None, None, None, None)
    tensor_feature.StarDepOrWeakDep = StarDepOrWeakDep
    tensor_feature.bytes = bytes
    tensor_feature.strides = [0] * ndims_lim
    if strides is not None and len(strides) > 0:
        strides = strides[-ndims_lim:]
        tensor_feature.strides[-len(strides) :] = strides
    tensor_feature.strides = np.array(tensor_feature.strides)
    assert len(tensor_feature.strides) == ndims_lim
    tensor_feature.sizes = [0] * ndims_lim
    if sizes is not None and len(sizes) > 0:
        sizes = sizes[-ndims_lim:]
        tensor_feature.sizes[-len(sizes) :] = sizes
    tensor_feature.sizes = np.array(tensor_feature.sizes)
    assert len(tensor_feature.sizes) == ndims_lim
    tensor_feature.is_contiguous = is_contiguous
    tensor_feature.is_scalar = is_scalar
    tensor_feature.is_indirect = is_indirect
    return tensor_feature


def dep_list(deps, strides, sizes, total_bytes, rw_lim, ndims_lim):
    res = list()
    # sort the tensors by bytes in descending order
    rw_list = sorted(
        zip(deps, strides, sizes, total_bytes), key=lambda x: x[-1], reverse=True
    )
    for dep, strides, sizes, bytes in rw_list[:rw_lim]:
        isStarDepOrWeakDep = isinstance(dep, (StarDep, WeakDep))
        dep_feature = make_dep_feature(
            ndims_lim=ndims_lim,
            StarDepOrWeakDep=isStarDepOrWeakDep,
            bytes=bytes,
            strides=strides,
            sizes=sizes,
            is_contiguous=dep.is_contiguous() if not isStarDepOrWeakDep else False,
            is_scalar=dep.is_scalar() if not isStarDepOrWeakDep else False,
            is_indirect=dep.is_indirect() if not isStarDepOrWeakDep else False,
        )
        if strides is not None and sizes is not None:
            assert len(strides) == len(sizes)
            for size_ in sizes:
                assert isinstance(size_, int)
            for stride in strides:
                assert isinstance(stride, int)
        res.append(dep_feature)
    for _ in range(rw_lim - len(res)):
        res.append(make_dep_feature(ndims_lim=ndims_lim))
    return res


# model arch related


class ModelType(IntEnum):
    """Given a kernel K, and a set of configs C, we would like to build a model f
    that predict the rankings of C by their corresponding running time t.

    This is naturally a learning-to-rank problem, which has been studied for years in ML.
    We use X to represent the feature vector of (K, c). Two common ways of loss formulation are:

    - Pointwise: we let f directly predict the (probably normalized) running time t.
               For each (K, c, t), loss = MSE_loss/L1_loss(f(K, c), t).

    - Pairwise: we let f predict a score as a proxy for ordering for each pair of (K, c).
               For each pair of (K, c1, t1) and (K, c2, t2), loss = BCE_loss(f(K, c1) > f(K, c2), t1 > t2).

    We consider the following models:
    0. XGB_BASELINE: the baseline model that uses XGBoost
    1. NN_POINTWISE: the neural network model that uses pointwise loss
    2. NN_PAIRWISE: the neural network model that uses pairwise loss
    3. NN_PAIRWISE_SMALL: the neural network model that uses pairwise loss, but trained on a smaller dataset
    that only contains the max autotune configs

    The neural network models are implemented in AutotunerFFN class below.

    Currently, 0 and 1 perform similarly, and 2 and 3 perform worse.
    """

    XGB_BASELINE = 0
    NN_POINTWISE = 1
    NN_PAIRWISE = 2
    NN_PAIRWISE_SMALL = 3


class AutotunerFFN(nn.Module):
    """The neural network model we used for autotuning."""

    def __init__(self, model_cfg):
        super().__init__()

        self.kernel_category_embedding = torch.nn.Embedding(
            num_embeddings=model_cfg["kernel_category_cnt"],
            embedding_dim=model_cfg["kernel_category_embed_dim"],
        )
        self.num_of_loops_embedding = torch.nn.Embedding(
            num_embeddings=model_cfg["num_of_loops_cnt"],
            embedding_dim=model_cfg["num_of_loops_embed_dim"],
        )

        self.hidden_dim = [model_cfg["feature_dim"]] + model_cfg["hidden_dim"] + [1]
        self.num_layers = len(self.hidden_dim) - 1

        self.op_bag_ln = nn.ModuleList(
            [
                nn.Linear(1, model_cfg["op_embed_dim"])
                for i in range(model_cfg["op_cnt"])
            ]
        )

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
                for i in range(self.num_layers)
            ]
        )
        self.use_norm = model_cfg["use_norm"]
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim[i + 1]) for i in range(self.num_layers - 1)]
        )

        self.activation = model_cfg["activation"]

    def get_feature_groups(self, x, show_progress=False):
        """We separate out this function to speed up the feature extraction process during training.
        Much of the time is spent on the feature extraction process. We first extract these feature
        tensors and store them once for all the configs, and then we can just load them during training.
        """
        device = "cuda"

        def normalize(x, positive=True):
            if positive:
                return torch.log2(x + 1)
            return torch.log2(torch.abs(x) + 1) * torch.sign(x)

        x_kernel_category = np.array([i.kernel_feature.kernel_category for i in x])
        x_kernel_category = torch.tensor(x_kernel_category, device=device).long()

        x_num_of_loops = np.array([i.kernel_feature.num_of_loops for i in x])
        x_num_of_loops = torch.tensor(x_num_of_loops, device=device).long()

        x_op_vec = np.array([i.kernel_feature.op_vec for i in x])
        x_op_vec = torch.tensor(x_op_vec, device=device).float()

        x_size_hints = np.array([i.kernel_feature.size_hints for i in x])
        x_size_hints = torch.tensor(x_size_hints, device=device).float()
        x_size_hints = normalize(x_size_hints)

        def get_dep_feature_vec(x, attr):
            assert attr in ["read_deps", "write_deps"]
            StarOrWeakdep_list = list()
            bytes_list = list()
            strides_list = list()
            sizes_list = list()
            is_contiguous_list = list()
            is_scalar_list = list()
            is_indirect_list = list()

            for i in x if not show_progress else tqdm.tqdm(x):
                for j in getattr(i.kernel_feature, attr):
                    StarOrWeakdep_list.append(j.StarDepOrWeakDep)
                    bytes_list.append(j.bytes)
                    strides_list.append(j.strides)
                    sizes_list.append(j.sizes)
                    is_contiguous_list.append(j.is_contiguous)
                    is_scalar_list.append(j.is_scalar)
                    is_indirect_list.append(j.is_indirect)

            StarOrWeakdep_tensor = torch.tensor(StarOrWeakdep_list, device=device)

            bytes_tensor = normalize(torch.tensor(bytes_list, device=device))
            strides_tensor = normalize(
                torch.tensor(np.array(strides_list), device=device), positive=False
            )
            sizes_tensor = normalize(
                torch.tensor(np.array(sizes_list), device=device), positive=False
            )
            is_contiguous_tensor = torch.tensor(is_contiguous_list, device=device)
            is_scalar_tensor = torch.tensor(is_scalar_list, device=device)
            is_indirect_tensor = torch.tensor(is_indirect_list, device=device)

            return (
                torch.cat(
                    [
                        StarOrWeakdep_tensor.unsqueeze(1),
                        bytes_tensor.unsqueeze(1),
                        strides_tensor,
                        sizes_tensor,
                        is_contiguous_tensor.unsqueeze(1),
                        is_scalar_tensor.unsqueeze(1),
                        is_indirect_tensor.unsqueeze(1),
                    ],
                    dim=1,
                )
                .reshape(len(x), -1)
                .float()
            )

        x_read_deps = get_dep_feature_vec(x, "read_deps")
        x_write_deps = get_dep_feature_vec(x, "write_deps")

        rest_vec = np.array(
            [
                [i.XBLOCK, i.YBLOCK, i.RBLOCK, i.num_warps, i.num_stages]
                + list(i.kernel_feature.numels)
                for i in x
            ]
        )

        rest_vec = torch.tensor(rest_vec, device=device)
        rest_vec[:, 0:3] = normalize(rest_vec[:, 0:3])
        rest_vec[:, -3:] = normalize(rest_vec[:, -3:])

        return (
            x_kernel_category,
            x_num_of_loops,
            x_op_vec,
            x_size_hints,
            x_read_deps,
            x_write_deps,
            rest_vec,
        )

    def forward_(self, x_tuple):
        (
            x_kernel_category,
            x_num_of_loops,
            x_op_vec,
            x_size_hints,
            x_read_deps,
            x_write_deps,
            rest_vec,
        ) = x_tuple
        x_kernel_category = self.kernel_category_embedding(x_kernel_category)
        x_num_of_loops = self.num_of_loops_embedding(x_num_of_loops)
        x_op_vec = torch.cat(
            [
                self.op_bag_ln[i](x_op_vec[:, i].unsqueeze(1))
                for i in range(len(OP_DICT))
            ],
            dim=1,
        )

        x = torch.cat(
            [
                x_kernel_category,
                x_num_of_loops,
                x_op_vec,
                x_size_hints,
                x_read_deps,
                x_write_deps,
                rest_vec,
            ],
            dim=-1,
        ).float()

        if self.use_norm:
            for norm, layer in zip(self.norms, self.layers[:-1]):
                x = self.activation(norm(layer(x)))
        else:
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def forward(self, x):
        return self.forward_(self.get_feature_groups(x))


class Autotuner_FFN(AutotunerFFN):
    pass


def get_model(model_type: ModelType):
    if model_type == ModelType.XGB_BASELINE:
        import xgboost

        return xgboost.XGBRegressor(
            max_depth=15,
            learning_rate=0.2,
            n_estimators=200,
            tree_method="hist",
            predictor="cpu_predictor",
            eval_metric=["rmse", "mae"],
        )
    else:
        model_cfg = {
            "kernel_category_cnt": 3,
            "kernel_category_embed_dim": 32,
            "num_of_loops_cnt": 10,
            "num_of_loops_embed_dim": 32,
            "feature_dim": 443,
            "op_embed_dim": 2,
            "op_cnt": len(OP_DICT),
            "bool_embed_dim": 4,
            "use_norm": True,
            "activation": torch.nn.functional.tanh,
        }
        if model_type == ModelType.NN_POINTWISE:
            model_cfg["hidden_dim"] = [8192, 2048, 32]
            return AutotunerFFN(model_cfg)
        elif model_type == ModelType.NN_PAIRWISE:
            model_cfg["hidden_dim"] = [4096, 1024, 32]
            return AutotunerFFN(model_cfg)
        elif model_type == ModelType.NN_PAIRWISE_SMALL:
            model_cfg["hidden_dim"] = [8192, 64 * 4]
            model_cfg["use_norm"] = False
            model_cfg["activation"] = torch.nn.functional.leaky_relu
            return AutotunerFFN(model_cfg)
        else:
            raise AssertionError("Unknown model type")


# search space related
class AutotunerSpaceCategory(IntEnum):
    """The model can exlpore different search spaces. Currently we support the following spaces:

    0. MAX_AUTOTUNE_TOP1: the model will predict the best config in max autotune configs
    and send the top1 config to compile.
    1. MAX_AUTOTUNE_TOP2: all the same as above, but send the top2 configs to profile and
    2. RADIUS_1_TOP1: the model will predict the best config in the search space that is
    within radius 1 of the max autotune configs, and send the top1 config to compile.
    3. RADIUS_1_TOP2: all the same as above, but send the top2 configs to profile.
    """

    MAX_AUTOTUNE_TOP1 = 0
    MAX_AUTOTUNE_TOP2 = 1
    RADIUS_1_TOP1 = 2
    RADIUS_1_TOP2 = 3


def get_heuristic_configs(autotuner_dict):
    assert "heuristics" in autotuner_dict
    assert "meta" in autotuner_dict
    heuristics = autotuner_dict["heuristics"]
    size_hints = autotuner_dict["size_hints"]
    meta = autotuner_dict["meta"]
    if heuristics == "pointwise":
        assert "tile_hint" in autotuner_dict
        return pointwise_heuristic(
            size_hints=size_hints, meta=meta, tile_hint=autotuner_dict["tile_hint"]
        )
    elif heuristics == "reduction":
        assert "reduction_hint" in autotuner_dict
        return reduction_heuristic(
            size_hints=size_hints,
            reduction_hint=autotuner_dict["reduction_hint"],
            meta=meta,
        )
    elif heuristics == "persistent_reduction":
        assert "reduction_hint" in autotuner_dict
        return persistent_reduction_heuristic(
            size_hints=size_hints,
            reduction_hint=autotuner_dict["reduction_hint"],
            meta=meta,
        )
    raise ValueError(f"Unknown heuristics {heuristics}")


# This class is inherited from coordinate_descent_tuner.py
class SearchSpaceGenerator(CoordescTuner):
    def __init__(self, size_hints):
        super().__init__(is_mm=False, size_hints=size_hints)

    def explore_neighour(self, st_config, radius):
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = get_field(st_config, field)
            if old_value is None:
                continue
            candidate_values = self.get_neighbour_values(
                field,
                old_value,
                radius=radius,
                include_self=True,
            )
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)

        res = list()
        choices = itertools.product(*candidate_values_list)
        for choice in choices:
            assert len(choice) == len(effective_fields)
            candidate_config = copy.deepcopy(st_config)
            for new_val, field in zip(choice, effective_fields):
                set_field(candidate_config, field, new_val)
            res.append(candidate_config)
        return res

    def generate(self, configs, radius):
        res = list()
        for config_ in configs:
            res.extend(self.explore_neighour(config_, radius))
        return res


# AutotunerModel related


@functools.lru_cache(None)
def load_model(autotuner_path):
    log.debug("loading model, pid {pid}", extra={"pid": os.getpid()})
    autotuner_model = pickle.load(open(autotuner_path, "rb"))
    autotuner_model.prepare()
    return autotuner_model


class AutotunerModel:
    """Wrapper for the autotuner model. It can load the model from disk and
    predict the best configs given a set of configs.

    It provides a unified interface for the autotuner to use different models.
    """

    model_type: ModelType
    # We only consider the last N_DIMS_LIM dimensions of strides and sizes
    ndims_lim: int
    # We only consider the first READ_DEP_LIM reads and first WRITE_DEP_LIM writes
    read_dep_lim: int
    write_dep_lim: int

    def __init__(self, model_type, read_dep_lim=10, write_dep_lim=5, ndims_lim=6):
        self.model_type = model_type
        self.model = get_model(model_type)
        self.read_dep_lim = read_dep_lim
        self.write_dep_lim = write_dep_lim
        self.ndims_lim = ndims_lim

    def load(self, path):
        if self.model_type == ModelType.XGB_BASELINE:
            self.model.load_model(path)
        else:
            self.model.load_state_dict(torch.load(path))

    def prepare(self):
        if self.model_type != ModelType.XGB_BASELINE:
            self.model = self.model.to("cuda")
            self.model.eval()

    def score_(self, X):
        if self.model_type == ModelType.XGB_BASELINE:
            scores = self.model.predict(X) * -1
        else:
            scores = self.model.forward(X).squeeze().cpu().detach().numpy()
            if self.model_type == ModelType.NN_POINTWISE:
                scores = scores * -1
        return scores

    def score(self, configs, autotuner_raw_data):
        X = self.get_feature_vec(configs, autotuner_raw_data)
        indices = np.argsort(self.score_(X))
        return [configs[i] for i in indices]

    def predict(self, configs, autotuner_raw_data, autotuner_space):
        _, _, src_code, _ = autotuner_raw_data
        size_hints = get_size_hints(src_code)

        configs = unique_configs(configs)
        if autotuner_space in [
            AutotunerSpaceCategory.RADIUS_1_TOP1,
            AutotunerSpaceCategory.RADIUS_1_TOP2,
        ]:
            configs = unique_configs(
                SearchSpaceGenerator(size_hints).generate(configs, radius=1)
            )

        configs = self.score(configs, autotuner_raw_data)
        if autotuner_space in [
            AutotunerSpaceCategory.MAX_AUTOTUNE_TOP1,
            AutotunerSpaceCategory.RADIUS_1_TOP1,
        ]:
            return configs[:1]
        elif autotuner_space in [
            AutotunerSpaceCategory.MAX_AUTOTUNE_TOP2,
            AutotunerSpaceCategory.RADIUS_1_TOP2,
        ]:
            return configs[:2]
        else:
            raise AssertionError("Unknown autotuner space")

    def feature_vector_to_xgb_input(self, feature_vecs):
        xgb_input = list()
        for feature_vector in feature_vecs:
            xgb_input.append(
                [
                    feature_vector.kernel_feature.kernel_category,
                    feature_vector.kernel_feature.num_of_loops,
                ]
                + list(feature_vector.kernel_feature.op_vec)
                + list(feature_vector.kernel_feature.size_hints)
                + list(
                    itertools.chain.from_iterable(
                        [
                            [
                                tensor_feature.StarDepOrWeakDep,
                                tensor_feature.bytes,
                            ]
                            + list(tensor_feature.strides)
                            + list(tensor_feature.sizes)
                            + [
                                tensor_feature.is_contiguous,
                                tensor_feature.is_scalar,
                                tensor_feature.is_indirect,
                            ]
                            for tensor_feature in feature_vector.kernel_feature.read_deps
                            + feature_vector.kernel_feature.write_deps
                        ]
                    )
                )
                + [
                    feature_vector.XBLOCK,
                    feature_vector.YBLOCK,
                    feature_vector.RBLOCK,
                    feature_vector.num_warps,
                    feature_vector.num_stages,
                ]
                + list(feature_vector.kernel_feature.numels)
            )
        return np.array(xgb_input)

    def get_feature_vec(self, configs, autotuner_raw_data):
        (
            (reads, writes, strides, sizes, total_bytes),
            node_read_writes,
            src_code,
            autotuner_dict,
        ) = autotuner_raw_data

        kernel_feature = KernelFeatureVector(None, None, None, None, None, None, None)
        # Get the kernel category
        kernel_feature.kernel_category = get_kernel_category(src_code)
        if kernel_feature.kernel_category is None:
            return None

        # Get the number of loops
        if kernel_feature.kernel_category is KernelCategory.REDUCTION:
            kernel_feature.num_of_loops = get_number_of_loops(src_code)
        else:
            kernel_feature.num_of_loops = 0

        # Get the op dict
        op_counts = node_read_writes.op_counts
        op_bag = dict()
        for op in sorted(op_counts.keys()):
            assert op in OP_DICT, "Unknown op: " + op
            op_bag[OP_DICT[op]] = op_counts[op]
        kernel_feature.op_vec = [op_bag.get(i, 0) for i in range(len(OP_DICT))]

        # Get the tilings
        kernel_feature.numels = np.array(get_tiling(src_code))

        # Get the size hints
        # TODO: fix this feature
        kernel_feature.size_hints = np.array([1] * 2)
        size_hints = get_size_hints(src_code)
        kernel_feature.size_hints[: len(size_hints)] = size_hints

        # Get the input tensors and output tensors
        kernel_feature.read_deps = dep_list(
            reads,
            strides[: len(reads)],
            sizes[: len(reads)],
            total_bytes[: len(reads)],
            self.read_dep_lim,
            self.ndims_lim,
        )
        kernel_feature.write_deps = dep_list(
            writes,
            strides[len(reads) :],
            sizes[len(reads) :],
            total_bytes[len(reads) :],
            self.write_dep_lim,
            self.ndims_lim,
        )

        X = list()
        for config_ in configs:
            feature_vector = FeatureVector(None, None, None, None, None, None)
            feature_vector.kernel_feature = kernel_feature
            feature_vector.XBLOCK = config_.kwargs.get("XBLOCK", 1)
            feature_vector.YBLOCK = config_.kwargs.get("YBLOCK", 1)
            feature_vector.RBLOCK = config_.kwargs.get("RBLOCK", 1)
            feature_vector.num_warps = config_.num_warps
            feature_vector.num_stages = config_.num_stages
            X.append(feature_vector)

        if self.model_type == ModelType.XGB_BASELINE:
            return self.feature_vector_to_xgb_input(X)
        else:
            return X


def autotuner_predict(autotuner_raw_data, autotuner_path):
    """The main function of the autotuner. It takes the raw data of the kernel
    and the path to the autotuner model. It extracts feature vectors from the raw data
    construct the search space and predict the best configs.
    """
    autotuner_dict = autotuner_raw_data.autotuner_dict
    src = autotuner_raw_data.src_code
    # get max autotune heursitic configs
    # we need to turn on max_autotune_pointwise to get these configs
    config.max_autotune_pointwise = True
    configs, _ = get_heuristic_configs(autotuner_dict)
    if len(configs) == 1:
        return configs[:1]

    no_X = src.find("XBLOCK: tl.constexpr") != -1
    no_Y = src.find("YBLOCK: tl.constexpr") != -1
    no_R = src.find("RBLOCK: tl.constexpr") != -1

    for tconfig in configs:
        if no_X:
            tconfig.kwargs.pop("XBLOCK", None)
        if no_Y:
            tconfig.kwargs.pop("YBLOCK", None)
        if no_R:
            tconfig.kwargs.pop("RBLOCK", None)

    autotuner = load_model(autotuner_path)
    assert isinstance(autotuner, AutotunerModel)

    # for cfg in configs:
    #     print(cfg.kwargs, cfg.num_stages, cfg.num_warps)
    #     print(autotuner_raw_data)
    #     print(autotuner.get_feature_vec([cfg], autotuner_raw_data))

    predicted_configs = autotuner.predict(
        configs, autotuner_raw_data, config.triton.autotuner_space
    )
    return predicted_configs
