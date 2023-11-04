__all__ = ["get_meta"]

import inspect
import re
import warnings
from typing import Any, Dict

import torch
from torch.testing import make_tensor


def get_meta(op, key, device_name=None, version=(0, torch.float16, 0.5), exact=False):
    """Return triton kernel meta parameters of the specified op and its inputs key.

    Parameters
    ----------
    op (str): The name of an operation that implementation uses meta parameters.
    key (tuple): A tuple of op input parameters, e.g. shapes, etc.
    device_name (optional, str): The name of a device for which op
      parameters are provided.
    version (optional, hashable): Specifies the version of parameters.
    exact (optional, bool): When True, the returned data (if
      available) corresponds exactly to the specified device_name and
      version information. Otherwise, if the corresponding data is not
      available but there exists a data set that is computed for a
      similar GPU device, then this data set will be returned.

    Returns
    -------
    result (dict): The requested mapping of parameter names and
      values, or None when no data is available.
    """
    if device_name is None:
        device_name = torch.cuda.get_device_name()
    op_data = _operation_device_version_data.get((op, device_name, version))
    if op_data is None and not exact:
        # A lack of op data could be due to using a (slightly)
        # different GPU model compared to a model for which optimal
        # meta parameters have been computed. In the following we'll
        # assume that there is a set of GPU models that all have
        # a similar set of optimal meta parameters.
        if re.match(r"NVIDIA A100[^\d]", device_name) is not None:
            device_name = "NVIDIA A100-SXM4-80GB"
        else:
            return
        op_data = _operation_device_version_data.get((op, device_name, version))
    if op_data is None:
        return
    values = op_data.get(key)
    if values is not None:
        if op == "scatter_mm":
            names = (
                "GROUP_SIZE",
                "SPLIT_N",
                "TILE_M",
                "TILE_N",
                "num_stages",
                "num_warps",
            )
            return dict(zip(names, values))
        elif op == "bsr_dense_mm":
            return dict(zip(("GROUP_SIZE_ROW", "num_stages", "num_warps"), values))

        raise NotImplementedError(f"names for {op=}")


def update(op, device_name, version, key, value):
    """Update the db of op parameters."""
    if (op, device_name, version) in _operation_device_version_data:
        if _operation_device_version_data[op, device_name, version].get(key) == value:
            return
        _operation_device_version_data[op, device_name, version][key] = value
    else:
        _operation_device_version_data[op, device_name, version] = {key: value}


def dump():
    """Store the current runtime db state to the module file."""
    current_file = inspect.getfile(dump)
    f = open(current_file)
    current_content = f.read()
    f.close()
    begin_data_str = "# BEGIN GENERATED DATA\n"
    begin_data_index = current_content.find(begin_data_str)
    end_data_index = current_content.find("    # END GENERATED DATA\n")
    if begin_data_index == -1 or end_data_index == -1:
        warnings.warn(
            f"{current_file} cannot be updated:"
            " BEGIN/END GENERATED DATA comment blocks appear to be corrupted"
        )
        return
    part1 = current_content[: begin_data_index + len(begin_data_str)]
    part2 = current_content[end_data_index:]
    data_part = []
    for op_key in sorted(_operation_device_version_data):
        data_part.append("    " + repr(op_key).replace("'", '"') + ": {")
        op_data = _operation_device_version_data[op_key]
        for key in sorted(op_data):
            data_part.append(f"        {key}: {op_data[key]},")
        data_part.append("    },")
    new_content = part1 + "\n".join(data_part) + "\n" + part2
    if current_content != new_content:
        f = open(current_file, "w")
        f.write(new_content)
        f.close()


def minimize(target_func, initial_parameters, step_func):
    """Find a dict of parameters that minimizes the target function using
    the initial dict of parameters and a step function that progresses
    a specified parameter in a dict of parameters.

    Parameters
    ----------
    target_func (callable): a functional with the signature
      ``target_func(parameters: dict) -> float``
    initial_parameters (dict): a set of parameters used as an initial
      value to the minimization process.
    step_func (callable): a functional with the signature
      ``step_func(parameter_name:str, parameter_value:int, direction:int, parameters:dict) -> int``
      that increments or decrements (when ``direction`` is positive or
      negative, respectively) the parameter with given name and value.
      When return value is equal to ``parameter_value``, it means that
      no step along the given direction can be made.

    Returns
    -------
    parameters (dict): a set of parameters that minimizes the target
      function.
    speedup_incr (float): a speedup change given in percentage
    """

    def to_key(parameters):
        return tuple(parameters[k] for k in sorted(parameters))

    def from_key(key, parameters):
        return dict(zip(sorted(parameters), key))

    all_values = dict()
    parameters = initial_parameters
    try:
        initial_target = target_func(parameters)
    except Exception as msg:
        print(f"{parameters=} lead to failure: {msg}. Skipping.")
        return parameters, -1
    all_values[to_key(parameters)] = initial_target

    while True:
        current_key = to_key(parameters)
        minimizer_target = all_values[current_key]
        minimizer_key = current_key
        new_minimizer = False
        for name in parameters:
            value = parameters[name]
            for direction in [1, -1]:
                next_value = step_func(name, value, direction, parameters)
                if next_value == value:
                    continue
                next_parameters = parameters.copy()
                next_parameters[name] = next_value
                next_key = to_key(next_parameters)
                if next_key in all_values:
                    continue
                try:
                    next_target = target_func(next_parameters)
                except Exception as msg:
                    all_values[next_key] = str(msg)
                    print(f"{next_parameters=} lead to failure: {msg}. Skipping.")
                    continue
                all_values[next_key] = next_target
                if next_target < minimizer_target:
                    minimizer_target = next_target
                    minimizer_key = next_key
                    new_minimizer = True
        if new_minimizer:
            parameters = from_key(minimizer_key, parameters)
        else:
            speedup_incr = (1 - minimizer_target / initial_target) * 100
            return parameters, speedup_incr


def create_blocked_tensor(B, M, N, blocksize, sparsity, dtype, device):
    assert (
        sparsity <= 1.0 and sparsity >= 0.0
    ), "sparsity should be a value between 0 and 1"
    assert M % blocksize[0] == 0
    assert N % blocksize[1] == 0
    shape = (B, M // blocksize[0], N // blocksize[1])[int(B == 0) :]
    A = torch.bernoulli(torch.full(shape, 1 - sparsity, dtype=dtype, device=device))
    expected_nnz = int((1 - sparsity) * M * N / (blocksize[0] * blocksize[1]))
    nonzero_indices = A.flatten().nonzero()
    actual_nnz = nonzero_indices.shape[0]
    if actual_nnz > expected_nnz:
        selected_nonzeros = torch.randperm(actual_nnz)[: actual_nnz - expected_nnz]
        A.flatten()[nonzero_indices[selected_nonzeros]] = 0
    elif actual_nnz < expected_nnz:
        zero_indices = (A == 0).flatten().nonzero()
        selected_zeros = torch.randperm(zero_indices.shape[0])[
            : expected_nnz - actual_nnz
        ]
        A.flatten()[zero_indices[selected_zeros]] = 1
    A = torch.repeat_interleave(A, blocksize[0], dim=-2)
    A = torch.repeat_interleave(A, blocksize[1], dim=-1)
    return A


def optimize_scatter_mm(
    m, k, n, bm, bk, dtype=torch.float16, device="cuda", sparsity=0.5, force=False
):
    import triton

    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data

    key = (m, k, n, bm, bk)
    version = (0, dtype, sparsity)

    initial_meta = get_meta("scatter_mm", key, version=version, exact=True)

    if initial_meta is None:
        initial_meta = get_meta("scatter_mm", key, version=(0, dtype, 0.5), exact=True)
        if initial_meta is None:
            initial_meta = dict(
                GROUP_SIZE=1, TILE_M=16, TILE_N=16, SPLIT_N=1, num_warps=1, num_stages=1
            )
    elif not force:
        return

    print(f"{m, k, n, bm, bk=}")
    torch.manual_seed(0)
    bsr = create_blocked_tensor(
        0, m, k, (bm, bk), sparsity, dtype, device
    ).to_sparse_bsr((bm, bk))
    dense = make_tensor(k, n, dtype=dtype, device=device)

    def bench(meta, bsr=bsr, dense=dense):
        indices_data = bsr_scatter_mm_indices_data(
            bsr, dense, indices_format="bsr_strided_mm_compressed", **meta
        )

        def test_func():
            return bsr_scatter_mm(bsr, dense, indices_data=indices_data)

        ms, ms_min, ms_max = triton.testing.do_bench(
            test_func, warmup=500, rep=100, fast_flush=False
        )

        return ms

    def step_meta_parameter(name, value, direction, meta, m=m, n=n, k=k, bm=bm, bk=bk):
        # return next value in positive or negative direction, or
        # input value if the step will result an invalid
        # value. The input value is assumed to be valid.
        is_log = name in {"SPLIT_N", "TILE_M", "TILE_N", "num_warps"}
        min_value = dict(
            SPLIT_N=1, TILE_M=16, TILE_N=16, num_warps=1, num_stages=1, GROUP_SIZE=1
        )[name]
        max_value = dict(
            SPLIT_N=n // meta["TILE_N"], TILE_M=bm, TILE_N=n // meta["SPLIT_N"]
        ).get(name)
        value_step = dict(
            SPLIT_N=2, TILE_M=2, TILE_N=2, num_warps=2, num_stages=1, GROUP_SIZE=1
        )[name]
        if is_log:
            next_value = value * value_step if direction > 0 else value // value_step
        else:
            next_value = value + value_step if direction > 0 else value - value_step
        if min_value is not None:
            next_value = max(next_value, min_value)
        if max_value is not None:
            next_value = min(next_value, max_value)
        if name == "SPLIT_N" and n % next_value != 0:
            return value
        return next_value

    meta, speedup = minimize(bench, initial_meta, step_meta_parameter)
    if speedup < 3 and 0:
        # don't bother updating parameters when the speed up change is less than 3 %
        return
    print(f"{meta=} {speedup=:.1f} %")
    device_name = torch.cuda.get_device_name()

    update(
        "scatter_mm", device_name, version, key, tuple(meta[k] for k in sorted(meta))
    )


def optimize_bsr_dense_mm(
    m, k, n, bm, bk, dtype=torch.float16, device="cuda", sparsity=0.5, force=False
):
    import triton

    from torch.sparse._triton_ops import bsr_dense_mm

    key = (m, k, n, bm, bk)
    version = (0, dtype, sparsity)

    initial_meta = get_meta("bsr_dense_mm", key, version=version, exact=True)

    if initial_meta is None:
        initial_meta = get_meta(
            "bsr_dense_mm", key, version=(0, dtype, 0.5), exact=True
        )
        if initial_meta is None:
            initial_meta = dict(GROUP_SIZE_ROW=1, num_stages=1, num_warps=1)
    elif not force:
        return

    print(f"{m, k, n, bm, bk=}")
    torch.manual_seed(0)
    bsr = create_blocked_tensor(
        0, m, k, (bm, bk), sparsity, dtype, device
    ).to_sparse_bsr((bm, bk))
    dense = make_tensor(k, n, dtype=dtype, device=device)

    def bench(meta, bsr=bsr, dense=dense):
        def test_func():
            return bsr_dense_mm(bsr, dense, meta=meta)

        ms, ms_min, ms_max = triton.testing.do_bench(
            test_func, warmup=500, rep=100, fast_flush=False
        )

        return ms

    def step_meta_parameter(name, value, direction, meta, m=m, n=n, k=k, bm=bm, bk=bk):
        # return next value in positive or negative direction, or
        # input value if the step will result an invalid
        # value. The input value is assumed to be valid.
        is_log = name in {"num_warps"}
        min_value = dict(num_warps=1, num_stages=1, GROUP_SIZE_ROW=1)[name]
        max_value = dict().get(name)
        value_step = dict(num_warps=2, num_stages=1, GROUP_SIZE_ROW=1)[name]
        if is_log:
            next_value = value * value_step if direction > 0 else value // value_step
        else:
            next_value = value + value_step if direction > 0 else value - value_step
        if min_value is not None:
            next_value = max(next_value, min_value)
        if max_value is not None:
            next_value = min(next_value, max_value)
        return next_value

    meta, speedup = minimize(bench, initial_meta, step_meta_parameter)
    if speedup < 3 and 0:
        # don't bother updating parameters when the speed up change is less than 3 %
        return
    print(f"{meta=} {speedup=:.1f} %")
    device_name = torch.cuda.get_device_name()

    update(
        "bsr_dense_mm", device_name, version, key, tuple(meta[k] for k in sorted(meta))
    )


def main(op="scatter_mm", force=False):
    import itertools

    dtype = torch.float16
    sizes_lst = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    shapes_lst = [(sz, sz) for sz in sizes_lst[:-3]]
    blocksize_lst = [(16, 16), (32, 32), (64, 64), (128, 128)]
    sparsity_lst = [0.5, 0.7, 0.3][:1]
    for sparsity in sparsity_lst:
        print(f"{sparsity=}")
        try:
            for (M, K), N, (BM, BK) in itertools.product(
                shapes_lst, sizes_lst, blocksize_lst
            ):
                if op == "scatter_mm":
                    optimize_scatter_mm(M, K, N, BM, BK, force=force, sparsity=sparsity)
                elif op == "bsr_dense_mm":
                    optimize_bsr_dense_mm(
                        M, K, N, BM, BK, force=force, sparsity=sparsity
                    )
                else:
                    raise NotImplementedError(op)
        except KeyboardInterrupt:
            break
        except Exception as msg:
            dump()
            print(msg)
    dump()

    if 0:
        # Check performance dependence on sparsity and apply
        # adjustments when differences are noticable (more than 10%).
        #
        # When using NVIDIA A100 GPU, the performance dependence on
        # sparsity is insignificant (0 % ... 10 %) for majority of
        # shapes/blocksizes combinations. However, for a very few
        # specific size combinations, the effect of sparsity on
        # performance can be up to 20 %.
        for (M, K), N, (BM, BK) in itertools.product(
            shapes_lst, sizes_lst, blocksize_lst
        ):
            meta_lst: list = []
            key = (M, K, N, BM, BK)
            for sparsity1 in sparsity_lst:
                torch.manual_seed(0)
                bsr = create_blocked_tensor(
                    0, M, K, (BM, BK), sparsity1, dtype, device="cuda"
                ).to_sparse_bsr((BM, BK))
                dense = make_tensor(K, N, dtype=dtype, device="cuda")
                meta_lst = []
                for sparsity in sparsity_lst:
                    meta = get_meta(op, key, version=(0, dtype, sparsity), exact=True)
                    if meta is None:
                        continue

                    def bench(meta, bsr=bsr, dense=dense):
                        import triton

                        if op == "scatter_mm":
                            from torch.sparse._triton_ops import (
                                bsr_scatter_mm,
                                bsr_scatter_mm_indices_data,
                            )

                            indices_data = bsr_scatter_mm_indices_data(
                                bsr,
                                dense,
                                indices_format="bsr_strided_mm_compressed",
                                **meta,
                            )

                            def test_func():
                                return bsr_scatter_mm(
                                    bsr, dense, indices_data=indices_data
                                )

                        elif op == "bsr_dense_mm":
                            from torch.sparse._triton_ops import bsr_dense_mm

                            def test_func():
                                return bsr_dense_mm(bsr, dense, meta=meta)

                        else:
                            raise NotImplementedError(op)

                        ms, ms_min, ms_max = triton.testing.do_bench(
                            test_func, warmup=500, rep=100, fast_flush=False
                        )

                        return ms

                    meta_lst.append(
                        (bench(meta), sparsity, tuple(meta[k] for k in sorted(meta)))
                    )
                if not meta_lst:
                    continue
                meta_lst = sorted(meta_lst)
                index = [i for i, item in enumerate(meta_lst) if item[1] == sparsity1][
                    0
                ]
                if meta_lst[0][2] == meta_lst[index][2]:
                    continue
                speeddiff = (1 - meta_lst[index][0] / meta_lst[0][0]) * 100
                if abs(speeddiff) < 10:
                    continue

                print(sparsity1, index, key, meta_lst, speeddiff)

                if index > 0:
                    device_name = torch.cuda.get_device_name()
                    meta = get_meta(
                        op, key, version=(0, dtype, meta_lst[0][1]), exact=True
                    )
                    update(
                        op,
                        device_name,
                        (0, dtype, sparsity1),
                        key,
                        tuple(meta[k] for k in sorted(meta)),
                    )
                    print("update")
                    dump()


_operation_device_version_data: Dict[Any, Dict] = {
    # Warning: the data in between the BEGIN/END DATA comment lines
    # below is generated. It can be updated either manually or via
    # calling dump function defined above.
    #
    # Legend [op: key -> data]:
    #   scatter_mm : M, K, N, Ms, Ks -> GROUP_SIZE, SPLIT_N, TILE_M, TILE_N, num_stages, num_warps
    #   bsr_dense_mm : M, K, N, Ms, Ks -> GROUP_SIZE_ROW, num_stages, num_warps
    #
    # BEGIN GENERATED DATA
    ("bsr_dense_mm", "NVIDIA A100-SXM4-80GB", (0, torch.float16, 0.3)): {
        (256, 256, 256, 16, 16): (5, 1, 2),
        (256, 256, 256, 32, 32): (4, 3, 4),
        (256, 256, 256, 64, 64): (3, 3, 4),
        (256, 256, 256, 128, 128): (4, 2, 8),
        (256, 256, 512, 16, 16): (4, 1, 4),
        (256, 256, 512, 32, 32): (4, 1, 4),
        (256, 256, 512, 64, 64): (4, 3, 8),
        (256, 256, 512, 128, 128): (4, 2, 8),
        (256, 256, 1024, 16, 16): (4, 1, 2),
        (256, 256, 1024, 32, 32): (4, 3, 4),
        (256, 256, 1024, 64, 64): (5, 3, 4),
        (256, 256, 1024, 128, 128): (4, 2, 8),
        (256, 256, 2048, 16, 16): (5, 1, 2),
        (256, 256, 2048, 32, 32): (5, 2, 4),
        (256, 256, 2048, 64, 64): (4, 3, 4),
        (256, 256, 2048, 128, 128): (4, 2, 8),
        (256, 256, 4096, 16, 16): (4, 1, 1),
        (256, 256, 4096, 32, 32): (4, 2, 4),
        (256, 256, 4096, 64, 64): (5, 3, 4),
        (256, 256, 4096, 128, 128): (4, 2, 8),
        (256, 256, 8192, 16, 16): (4, 3, 1),
        (256, 256, 8192, 32, 32): (4, 3, 2),
        (256, 256, 8192, 64, 64): (4, 2, 4),
        (256, 256, 8192, 128, 128): (4, 1, 4),
        (256, 256, 16384, 16, 16): (5, 3, 1),
        (256, 256, 16384, 32, 32): (3, 2, 2),
        (256, 256, 16384, 64, 64): (4, 2, 4),
        (256, 256, 16384, 128, 128): (4, 1, 4),
        (256, 256, 32768, 16, 16): (5, 3, 1),
        (256, 256, 32768, 32, 32): (5, 3, 2),
        (256, 256, 32768, 64, 64): (4, 2, 4),
        (256, 256, 32768, 128, 128): (4, 1, 4),
        (256, 256, 65536, 16, 16): (4, 3, 1),
        (256, 256, 65536, 32, 32): (3, 3, 1),
        (256, 256, 65536, 64, 64): (5, 2, 4),
        (256, 256, 65536, 128, 128): (4, 1, 4),
        (256, 256, 131072, 16, 16): (1, 1, 2),
        (256, 256, 131072, 32, 32): (4, 2, 2),
        (256, 256, 131072, 64, 64): (4, 2, 4),
        (256, 256, 131072, 128, 128): (4, 1, 4),
        (512, 512, 256, 16, 16): (4, 1, 2),
        (512, 512, 256, 32, 32): (4, 1, 4),
        (512, 512, 256, 64, 64): (5, 3, 4),
        (512, 512, 256, 128, 128): (4, 2, 8),
        (512, 512, 512, 16, 16): (4, 1, 2),
        (512, 512, 512, 32, 32): (5, 1, 4),
        (512, 512, 512, 64, 64): (4, 3, 4),
        (512, 512, 512, 128, 128): (4, 2, 8),
        (512, 512, 1024, 16, 16): (4, 3, 1),
        (512, 512, 1024, 32, 32): (4, 1, 4),
        (512, 512, 1024, 64, 64): (5, 4, 4),
        (512, 512, 1024, 128, 128): (4, 2, 8),
        (512, 512, 2048, 16, 16): (4, 3, 1),
        (512, 512, 2048, 32, 32): (4, 3, 2),
        (512, 512, 2048, 64, 64): (2, 3, 4),
        (512, 512, 2048, 128, 128): (3, 2, 8),
        (512, 512, 4096, 16, 16): (4, 3, 1),
        (512, 512, 4096, 32, 32): (4, 4, 1),
        (512, 512, 4096, 64, 64): (4, 3, 4),
        (512, 512, 4096, 128, 128): (4, 2, 8),
        (512, 512, 8192, 16, 16): (4, 3, 1),
        (512, 512, 8192, 32, 32): (5, 3, 1),
        (512, 512, 8192, 64, 64): (5, 2, 4),
        (512, 512, 8192, 128, 128): (3, 1, 4),
        (512, 512, 16384, 16, 16): (4, 3, 1),
        (512, 512, 16384, 32, 32): (4, 3, 1),
        (512, 512, 16384, 64, 64): (4, 3, 2),
        (512, 512, 16384, 128, 128): (4, 1, 4),
        (512, 512, 32768, 16, 16): (2, 2, 1),
        (512, 512, 32768, 32, 32): (5, 3, 1),
        (512, 512, 32768, 64, 64): (4, 3, 2),
        (512, 512, 32768, 128, 128): (6, 1, 4),
        (512, 512, 65536, 16, 16): (5, 2, 1),
        (512, 512, 65536, 32, 32): (5, 3, 1),
        (512, 512, 65536, 64, 64): (4, 3, 2),
        (512, 512, 65536, 128, 128): (5, 1, 4),
        (512, 512, 131072, 16, 16): (4, 1, 4),
        (512, 512, 131072, 32, 32): (4, 2, 2),
        (512, 512, 131072, 64, 64): (4, 3, 2),
        (512, 512, 131072, 128, 128): (4, 1, 4),
        (1024, 1024, 256, 16, 16): (4, 4, 1),
        (1024, 1024, 256, 32, 32): (5, 1, 4),
        (1024, 1024, 256, 64, 64): (4, 4, 4),
        (1024, 1024, 256, 128, 128): (4, 2, 8),
        (1024, 1024, 512, 16, 16): (4, 3, 1),
        (1024, 1024, 512, 32, 32): (5, 4, 1),
        (1024, 1024, 512, 64, 64): (4, 3, 4),
        (1024, 1024, 512, 128, 128): (3, 2, 8),
        (1024, 1024, 1024, 16, 16): (4, 4, 1),
        (1024, 1024, 1024, 32, 32): (1, 3, 1),
        (1024, 1024, 1024, 64, 64): (1, 3, 4),
        (1024, 1024, 1024, 128, 128): (1, 2, 8),
        (1024, 1024, 2048, 16, 16): (5, 3, 1),
        (1024, 1024, 2048, 32, 32): (4, 4, 1),
        (1024, 1024, 2048, 64, 64): (3, 3, 2),
        (1024, 1024, 2048, 128, 128): (4, 2, 8),
        (1024, 1024, 4096, 16, 16): (4, 3, 1),
        (1024, 1024, 4096, 32, 32): (5, 3, 1),
        (1024, 1024, 4096, 64, 64): (5, 3, 4),
        (1024, 1024, 4096, 128, 128): (4, 2, 8),
        (1024, 1024, 8192, 16, 16): (4, 3, 1),
        (1024, 1024, 8192, 32, 32): (4, 3, 1),
        (1024, 1024, 8192, 64, 64): (4, 3, 2),
        (1024, 1024, 8192, 128, 128): (4, 1, 4),
        (1024, 1024, 16384, 16, 16): (4, 2, 1),
        (1024, 1024, 16384, 32, 32): (4, 3, 1),
        (1024, 1024, 16384, 64, 64): (4, 3, 2),
        (1024, 1024, 16384, 128, 128): (4, 1, 4),
        (1024, 1024, 32768, 16, 16): (4, 2, 1),
        (1024, 1024, 32768, 32, 32): (5, 3, 1),
        (1024, 1024, 32768, 64, 64): (5, 3, 2),
        (1024, 1024, 32768, 128, 128): (4, 1, 4),
        (1024, 1024, 65536, 16, 16): (8, 2, 1),
        (1024, 1024, 65536, 32, 32): (5, 3, 1),
        (1024, 1024, 65536, 64, 64): (5, 3, 2),
        (1024, 1024, 65536, 128, 128): (4, 1, 4),
        (1024, 1024, 131072, 16, 16): (1, 1, 4),
        (1024, 1024, 131072, 32, 32): (4, 2, 1),
        (1024, 1024, 131072, 64, 64): (5, 3, 2),
        (1024, 1024, 131072, 128, 128): (4, 1, 4),
        (2048, 2048, 256, 16, 16): (4, 4, 1),
        (2048, 2048, 256, 32, 32): (4, 4, 2),
        (2048, 2048, 256, 64, 64): (5, 4, 4),
        (2048, 2048, 256, 128, 128): (4, 2, 8),
        (2048, 2048, 512, 16, 16): (4, 4, 1),
        (2048, 2048, 512, 32, 32): (5, 3, 2),
        (2048, 2048, 512, 64, 64): (6, 3, 4),
        (2048, 2048, 512, 128, 128): (4, 2, 8),
        (2048, 2048, 1024, 16, 16): (5, 3, 1),
        (2048, 2048, 1024, 32, 32): (4, 3, 4),
        (2048, 2048, 1024, 64, 64): (5, 3, 4),
        (2048, 2048, 1024, 128, 128): (4, 2, 8),
        (2048, 2048, 2048, 16, 16): (2, 3, 1),
        (2048, 2048, 2048, 32, 32): (2, 4, 1),
        (2048, 2048, 2048, 64, 64): (3, 3, 2),
        (2048, 2048, 2048, 128, 128): (4, 2, 8),
        (2048, 2048, 4096, 16, 16): (4, 3, 1),
        (2048, 2048, 4096, 32, 32): (4, 4, 2),
        (2048, 2048, 4096, 64, 64): (4, 3, 2),
        (2048, 2048, 4096, 128, 128): (4, 1, 4),
        (2048, 2048, 8192, 16, 16): (6, 2, 1),
        (2048, 2048, 8192, 32, 32): (4, 4, 2),
        (2048, 2048, 8192, 64, 64): (4, 3, 2),
        (2048, 2048, 8192, 128, 128): (4, 1, 4),
        (2048, 2048, 16384, 16, 16): (6, 2, 1),
        (2048, 2048, 16384, 32, 32): (4, 4, 2),
        (2048, 2048, 16384, 64, 64): (4, 3, 2),
        (2048, 2048, 16384, 128, 128): (4, 1, 4),
        (2048, 2048, 32768, 16, 16): (4, 2, 1),
        (2048, 2048, 32768, 32, 32): (5, 4, 1),
        (2048, 2048, 32768, 64, 64): (5, 3, 2),
        (2048, 2048, 32768, 128, 128): (4, 1, 4),
        (2048, 2048, 65536, 16, 16): (4, 2, 1),
        (2048, 2048, 65536, 32, 32): (5, 5, 1),
        (2048, 2048, 65536, 64, 64): (5, 3, 2),
        (2048, 2048, 65536, 128, 128): (4, 1, 4),
        (2048, 2048, 131072, 16, 16): (4, 1, 4),
        (2048, 2048, 131072, 32, 32): (4, 1, 1),
        (2048, 2048, 131072, 64, 64): (5, 3, 2),
        (2048, 2048, 131072, 128, 128): (4, 1, 4),
        (4096, 4096, 256, 16, 16): (4, 5, 1),
        (4096, 4096, 256, 32, 32): (4, 3, 2),
        (4096, 4096, 256, 64, 64): (3, 3, 4),
        (4096, 4096, 256, 128, 128): (3, 2, 8),
        (4096, 4096, 512, 16, 16): (4, 4, 1),
        (4096, 4096, 512, 32, 32): (4, 3, 4),
        (4096, 4096, 512, 64, 64): (5, 3, 4),
        (4096, 4096, 512, 128, 128): (4, 2, 8),
        (4096, 4096, 1024, 16, 16): (3, 3, 1),
        (4096, 4096, 1024, 32, 32): (4, 4, 2),
        (4096, 4096, 1024, 64, 64): (3, 4, 4),
        (4096, 4096, 1024, 128, 128): (4, 2, 8),
        (4096, 4096, 2048, 16, 16): (4, 3, 1),
        (4096, 4096, 2048, 32, 32): (4, 4, 2),
        (4096, 4096, 2048, 64, 64): (3, 3, 2),
        (4096, 4096, 2048, 128, 128): (4, 1, 4),
        (4096, 4096, 4096, 16, 16): (1, 3, 1),
        (4096, 4096, 4096, 32, 32): (4, 4, 2),
        (4096, 4096, 4096, 64, 64): (1, 3, 2),
        (4096, 4096, 4096, 128, 128): (4, 1, 4),
        (4096, 4096, 8192, 16, 16): (4, 2, 1),
        (4096, 4096, 8192, 32, 32): (4, 4, 1),
        (4096, 4096, 8192, 64, 64): (4, 3, 2),
        (4096, 4096, 8192, 128, 128): (4, 1, 4),
        (4096, 4096, 16384, 16, 16): (1, 1, 1),
        (4096, 4096, 16384, 32, 32): (4, 4, 1),
        (4096, 4096, 16384, 64, 64): (4, 3, 2),
        (4096, 4096, 16384, 128, 128): (4, 1, 4),
        (4096, 4096, 32768, 16, 16): (4, 2, 1),
        (4096, 4096, 32768, 32, 32): (5, 3, 1),
        (4096, 4096, 32768, 64, 64): (5, 3, 2),
        (4096, 4096, 32768, 128, 128): (4, 1, 4),
        (4096, 4096, 65536, 16, 16): (4, 2, 1),
        (4096, 4096, 65536, 32, 32): (1, 2, 1),
        (4096, 4096, 65536, 64, 64): (3, 3, 2),
        (4096, 4096, 65536, 128, 128): (4, 1, 4),
        (4096, 4096, 131072, 16, 16): (1, 1, 4),
        (4096, 4096, 131072, 32, 32): (4, 2, 1),
        (4096, 4096, 131072, 64, 64): (5, 3, 2),
        (4096, 4096, 131072, 128, 128): (4, 1, 4),
        (8192, 8192, 256, 16, 16): (4, 5, 1),
        (8192, 8192, 256, 32, 32): (4, 3, 4),
        (8192, 8192, 256, 64, 64): (4, 3, 4),
        (8192, 8192, 256, 128, 128): (5, 1, 4),
        (8192, 8192, 512, 16, 16): (4, 6, 1),
        (8192, 8192, 512, 32, 32): (4, 4, 2),
        (8192, 8192, 512, 64, 64): (4, 4, 4),
        (8192, 8192, 512, 128, 128): (4, 2, 8),
        (8192, 8192, 1024, 16, 16): (4, 7, 1),
        (8192, 8192, 1024, 32, 32): (4, 4, 2),
        (8192, 8192, 1024, 64, 64): (4, 4, 4),
        (8192, 8192, 1024, 128, 128): (4, 1, 4),
        (8192, 8192, 2048, 16, 16): (4, 3, 1),
        (8192, 8192, 2048, 32, 32): (4, 4, 1),
        (8192, 8192, 2048, 64, 64): (4, 3, 2),
        (8192, 8192, 2048, 128, 128): (4, 1, 4),
        (8192, 8192, 4096, 16, 16): (4, 4, 1),
        (8192, 8192, 4096, 32, 32): (4, 4, 1),
        (8192, 8192, 4096, 64, 64): (4, 3, 2),
        (8192, 8192, 4096, 128, 128): (4, 1, 4),
        (8192, 8192, 8192, 16, 16): (4, 2, 1),
        (8192, 8192, 8192, 32, 32): (4, 4, 1),
        (8192, 8192, 8192, 64, 64): (4, 3, 2),
        (8192, 8192, 8192, 128, 128): (4, 1, 4),
        (8192, 8192, 16384, 16, 16): (4, 2, 1),
        (8192, 8192, 16384, 32, 32): (4, 4, 1),
        (8192, 8192, 16384, 64, 64): (4, 3, 2),
        (8192, 8192, 16384, 128, 128): (4, 1, 4),
        (8192, 8192, 32768, 16, 16): (4, 2, 1),
        (8192, 8192, 32768, 32, 32): (4, 4, 1),
        (8192, 8192, 32768, 64, 64): (4, 3, 2),
        (8192, 8192, 32768, 128, 128): (4, 1, 4),
        (8192, 8192, 65536, 16, 16): (4, 2, 1),
        (8192, 8192, 65536, 32, 32): (4, 4, 1),
        (8192, 8192, 65536, 64, 64): (4, 3, 2),
        (8192, 8192, 65536, 128, 128): (4, 1, 4),
        (8192, 8192, 131072, 16, 16): (4, 1, 4),
        (8192, 8192, 131072, 32, 32): (4, 2, 1),
        (8192, 8192, 131072, 64, 64): (4, 3, 2),
        (8192, 8192, 131072, 128, 128): (4, 1, 4),
        (16384, 16384, 256, 16, 16): (4, 4, 1),
        (16384, 16384, 256, 32, 32): (4, 4, 2),
        (16384, 16384, 256, 64, 64): (4, 4, 4),
        (16384, 16384, 256, 128, 128): (4, 2, 8),
        (16384, 16384, 512, 16, 16): (4, 3, 1),
        (16384, 16384, 512, 32, 32): (4, 4, 1),
        (16384, 16384, 512, 64, 64): (4, 4, 4),
        (16384, 16384, 512, 128, 128): (4, 2, 8),
        (16384, 16384, 1024, 16, 16): (4, 3, 1),
        (16384, 16384, 1024, 32, 32): (4, 4, 1),
        (16384, 16384, 1024, 64, 64): (4, 4, 4),
        (16384, 16384, 1024, 128, 128): (4, 1, 4),
        (16384, 16384, 2048, 16, 16): (4, 3, 1),
        (16384, 16384, 2048, 32, 32): (4, 4, 1),
        (16384, 16384, 2048, 64, 64): (4, 3, 2),
        (16384, 16384, 2048, 128, 128): (4, 1, 4),
        (16384, 16384, 4096, 16, 16): (4, 2, 1),
        (16384, 16384, 4096, 32, 32): (4, 5, 1),
        (16384, 16384, 4096, 64, 64): (4, 3, 2),
        (16384, 16384, 4096, 128, 128): (4, 1, 4),
        (16384, 16384, 8192, 16, 16): (4, 2, 1),
        (16384, 16384, 8192, 32, 32): (4, 4, 1),
        (16384, 16384, 8192, 64, 64): (4, 3, 2),
        (16384, 16384, 8192, 128, 128): (4, 1, 4),
        (16384, 16384, 16384, 16, 16): (4, 2, 1),
        (16384, 16384, 16384, 32, 32): (4, 6, 1),
        (16384, 16384, 16384, 64, 64): (4, 3, 2),
        (16384, 16384, 16384, 128, 128): (4, 1, 4),
        (16384, 16384, 32768, 16, 16): (4, 2, 1),
        (16384, 16384, 32768, 32, 32): (4, 6, 1),
        (16384, 16384, 32768, 64, 64): (4, 3, 2),
        (16384, 16384, 32768, 128, 128): (4, 1, 4),
        (16384, 16384, 65536, 16, 16): (4, 2, 1),
        (16384, 16384, 65536, 32, 32): (4, 2, 1),
        (16384, 16384, 65536, 64, 64): (4, 3, 2),
        (16384, 16384, 65536, 128, 128): (4, 1, 4),
        (16384, 16384, 131072, 16, 16): (4, 1, 4),
        (16384, 16384, 131072, 32, 32): (4, 2, 1),
        (16384, 16384, 131072, 64, 64): (4, 3, 2),
        (16384, 16384, 131072, 128, 128): (4, 1, 4),
    },
    ("bsr_dense_mm", "NVIDIA A100-SXM4-80GB", (0, torch.float16, 0.5)): {
        (256, 256, 256, 16, 16): (8, 1, 4),
        (256, 256, 256, 32, 32): (4, 3, 4),
        (256, 256, 256, 64, 64): (3, 3, 4),
        (256, 256, 256, 128, 128): (4, 2, 8),
        (256, 256, 512, 16, 16): (3, 1, 4),
        (256, 256, 512, 32, 32): (4, 1, 4),
        (256, 256, 512, 64, 64): (4, 3, 4),
        (256, 256, 512, 128, 128): (4, 2, 4),
        (256, 256, 1024, 16, 16): (4, 1, 2),
        (256, 256, 1024, 32, 32): (4, 3, 4),
        (256, 256, 1024, 64, 64): (4, 3, 4),
        (256, 256, 1024, 128, 128): (4, 2, 8),
        (256, 256, 2048, 16, 16): (4, 1, 1),
        (256, 256, 2048, 32, 32): (5, 1, 4),
        (256, 256, 2048, 64, 64): (4, 3, 4),
        (256, 256, 2048, 128, 128): (4, 2, 8),
        (256, 256, 4096, 16, 16): (4, 1, 1),
        (256, 256, 4096, 32, 32): (4, 1, 4),
        (256, 256, 4096, 64, 64): (4, 3, 4),
        (256, 256, 4096, 128, 128): (4, 2, 8),
        (256, 256, 8192, 16, 16): (5, 3, 1),
        (256, 256, 8192, 32, 32): (4, 3, 2),
        (256, 256, 8192, 64, 64): (4, 2, 4),
        (256, 256, 8192, 128, 128): (4, 1, 4),
        (256, 256, 16384, 16, 16): (5, 2, 1),
        (256, 256, 16384, 32, 32): (3, 2, 2),
        (256, 256, 16384, 64, 64): (4, 1, 4),
        (256, 256, 16384, 128, 128): (4, 1, 4),
        (256, 256, 32768, 16, 16): (5, 3, 1),
        (256, 256, 32768, 32, 32): (4, 3, 2),
        (256, 256, 32768, 64, 64): (4, 2, 4),
        (256, 256, 32768, 128, 128): (4, 1, 4),
        (256, 256, 65536, 16, 16): (5, 3, 1),
        (256, 256, 65536, 32, 32): (4, 3, 1),
        (256, 256, 65536, 64, 64): (5, 2, 4),
        (256, 256, 65536, 128, 128): (4, 1, 4),
        (256, 256, 131072, 16, 16): (4, 1, 2),
        (256, 256, 131072, 32, 32): (4, 2, 2),
        (256, 256, 131072, 64, 64): (4, 2, 4),
        (256, 256, 131072, 128, 128): (4, 1, 4),
        (512, 512, 256, 16, 16): (4, 1, 4),
        (512, 512, 256, 32, 32): (4, 1, 4),
        (512, 512, 256, 64, 64): (4, 3, 4),
        (512, 512, 256, 128, 128): (4, 2, 8),
        (512, 512, 512, 16, 16): (4, 1, 1),
        (512, 512, 512, 32, 32): (4, 1, 4),
        (512, 512, 512, 64, 64): (4, 3, 4),
        (512, 512, 512, 128, 128): (4, 2, 8),
        (512, 512, 1024, 16, 16): (5, 4, 1),
        (512, 512, 1024, 32, 32): (4, 1, 4),
        (512, 512, 1024, 64, 64): (4, 4, 4),
        (512, 512, 1024, 128, 128): (4, 2, 8),
        (512, 512, 2048, 16, 16): (4, 3, 1),
        (512, 512, 2048, 32, 32): (4, 3, 2),
        (512, 512, 2048, 64, 64): (3, 3, 4),
        (512, 512, 2048, 128, 128): (4, 2, 8),
        (512, 512, 4096, 16, 16): (4, 3, 1),
        (512, 512, 4096, 32, 32): (4, 4, 2),
        (512, 512, 4096, 64, 64): (4, 3, 4),
        (512, 512, 4096, 128, 128): (4, 2, 8),
        (512, 512, 8192, 16, 16): (4, 3, 1),
        (512, 512, 8192, 32, 32): (5, 3, 2),
        (512, 512, 8192, 64, 64): (5, 2, 4),
        (512, 512, 8192, 128, 128): (4, 1, 4),
        (512, 512, 16384, 16, 16): (4, 3, 1),
        (512, 512, 16384, 32, 32): (4, 3, 1),
        (512, 512, 16384, 64, 64): (4, 3, 4),
        (512, 512, 16384, 128, 128): (4, 1, 4),
        (512, 512, 32768, 16, 16): (4, 2, 1),
        (512, 512, 32768, 32, 32): (5, 3, 1),
        (512, 512, 32768, 64, 64): (4, 3, 2),
        (512, 512, 32768, 128, 128): (5, 1, 4),
        (512, 512, 65536, 16, 16): (4, 3, 1),
        (512, 512, 65536, 32, 32): (4, 3, 1),
        (512, 512, 65536, 64, 64): (4, 3, 2),
        (512, 512, 65536, 128, 128): (5, 1, 4),
        (512, 512, 131072, 16, 16): (4, 1, 4),
        (512, 512, 131072, 32, 32): (4, 2, 2),
        (512, 512, 131072, 64, 64): (4, 3, 2),
        (512, 512, 131072, 128, 128): (4, 1, 4),
        (1024, 1024, 256, 16, 16): (4, 4, 1),
        (1024, 1024, 256, 32, 32): (4, 1, 4),
        (1024, 1024, 256, 64, 64): (4, 4, 4),
        (1024, 1024, 256, 128, 128): (4, 2, 8),
        (1024, 1024, 512, 16, 16): (4, 4, 1),
        (1024, 1024, 512, 32, 32): (5, 4, 2),
        (1024, 1024, 512, 64, 64): (4, 3, 4),
        (1024, 1024, 512, 128, 128): (3, 2, 8),
        (1024, 1024, 1024, 16, 16): (5, 3, 1),
        (1024, 1024, 1024, 32, 32): (1, 3, 1),
        (1024, 1024, 1024, 64, 64): (1, 3, 2),
        (1024, 1024, 1024, 128, 128): (1, 2, 8),
        (1024, 1024, 2048, 16, 16): (4, 3, 1),
        (1024, 1024, 2048, 32, 32): (4, 4, 1),
        (1024, 1024, 2048, 64, 64): (3, 3, 4),
        (1024, 1024, 2048, 128, 128): (4, 2, 8),
        (1024, 1024, 4096, 16, 16): (4, 3, 1),
        (1024, 1024, 4096, 32, 32): (4, 3, 1),
        (1024, 1024, 4096, 64, 64): (4, 3, 4),
        (1024, 1024, 4096, 128, 128): (4, 2, 8),
        (1024, 1024, 8192, 16, 16): (4, 3, 1),
        (1024, 1024, 8192, 32, 32): (4, 3, 1),
        (1024, 1024, 8192, 64, 64): (4, 3, 4),
        (1024, 1024, 8192, 128, 128): (4, 1, 4),
        (1024, 1024, 16384, 16, 16): (4, 2, 1),
        (1024, 1024, 16384, 32, 32): (4, 3, 1),
        (1024, 1024, 16384, 64, 64): (4, 3, 2),
        (1024, 1024, 16384, 128, 128): (4, 1, 4),
        (1024, 1024, 32768, 16, 16): (4, 2, 1),
        (1024, 1024, 32768, 32, 32): (5, 3, 2),
        (1024, 1024, 32768, 64, 64): (5, 3, 2),
        (1024, 1024, 32768, 128, 128): (4, 1, 4),
        (1024, 1024, 65536, 16, 16): (4, 2, 1),
        (1024, 1024, 65536, 32, 32): (5, 3, 1),
        (1024, 1024, 65536, 64, 64): (5, 3, 2),
        (1024, 1024, 65536, 128, 128): (4, 1, 4),
        (1024, 1024, 131072, 16, 16): (4, 1, 4),
        (1024, 1024, 131072, 32, 32): (4, 2, 1),
        (1024, 1024, 131072, 64, 64): (5, 3, 2),
        (1024, 1024, 131072, 128, 128): (4, 1, 4),
        (2048, 2048, 256, 16, 16): (5, 4, 1),
        (2048, 2048, 256, 32, 32): (4, 4, 2),
        (2048, 2048, 256, 64, 64): (5, 3, 4),
        (2048, 2048, 256, 128, 128): (4, 2, 8),
        (2048, 2048, 512, 16, 16): (4, 4, 1),
        (2048, 2048, 512, 32, 32): (5, 3, 2),
        (2048, 2048, 512, 64, 64): (6, 3, 4),
        (2048, 2048, 512, 128, 128): (4, 2, 8),
        (2048, 2048, 1024, 16, 16): (4, 3, 1),
        (2048, 2048, 1024, 32, 32): (4, 3, 4),
        (2048, 2048, 1024, 64, 64): (5, 3, 4),
        (2048, 2048, 1024, 128, 128): (4, 2, 8),
        (2048, 2048, 2048, 16, 16): (3, 3, 1),
        (2048, 2048, 2048, 32, 32): (1, 4, 1),
        (2048, 2048, 2048, 64, 64): (3, 3, 2),
        (2048, 2048, 2048, 128, 128): (4, 2, 8),
        (2048, 2048, 4096, 16, 16): (4, 3, 1),
        (2048, 2048, 4096, 32, 32): (4, 4, 2),
        (2048, 2048, 4096, 64, 64): (4, 3, 2),
        (2048, 2048, 4096, 128, 128): (4, 1, 4),
        (2048, 2048, 8192, 16, 16): (4, 2, 1),
        (2048, 2048, 8192, 32, 32): (4, 4, 2),
        (2048, 2048, 8192, 64, 64): (4, 3, 2),
        (2048, 2048, 8192, 128, 128): (4, 1, 4),
        (2048, 2048, 16384, 16, 16): (4, 2, 1),
        (2048, 2048, 16384, 32, 32): (4, 4, 2),
        (2048, 2048, 16384, 64, 64): (4, 3, 2),
        (2048, 2048, 16384, 128, 128): (4, 1, 4),
        (2048, 2048, 32768, 16, 16): (6, 2, 1),
        (2048, 2048, 32768, 32, 32): (5, 4, 1),
        (2048, 2048, 32768, 64, 64): (5, 3, 2),
        (2048, 2048, 32768, 128, 128): (4, 1, 4),
        (2048, 2048, 65536, 16, 16): (3, 2, 1),
        (2048, 2048, 65536, 32, 32): (5, 4, 1),
        (2048, 2048, 65536, 64, 64): (4, 3, 2),
        (2048, 2048, 65536, 128, 128): (4, 1, 4),
        (2048, 2048, 131072, 16, 16): (4, 1, 4),
        (2048, 2048, 131072, 32, 32): (4, 1, 1),
        (2048, 2048, 131072, 64, 64): (5, 3, 2),
        (2048, 2048, 131072, 128, 128): (4, 1, 4),
        (4096, 4096, 256, 16, 16): (4, 4, 1),
        (4096, 4096, 256, 32, 32): (4, 3, 2),
        (4096, 4096, 256, 64, 64): (3, 3, 4),
        (4096, 4096, 256, 128, 128): (3, 2, 8),
        (4096, 4096, 512, 16, 16): (1, 3, 1),
        (4096, 4096, 512, 32, 32): (4, 3, 4),
        (4096, 4096, 512, 64, 64): (6, 3, 4),
        (4096, 4096, 512, 128, 128): (4, 2, 8),
        (4096, 4096, 1024, 16, 16): (3, 3, 1),
        (4096, 4096, 1024, 32, 32): (4, 4, 2),
        (4096, 4096, 1024, 64, 64): (4, 4, 4),
        (4096, 4096, 1024, 128, 128): (4, 2, 8),
        (4096, 4096, 2048, 16, 16): (1, 3, 1),
        (4096, 4096, 2048, 32, 32): (3, 4, 2),
        (4096, 4096, 2048, 64, 64): (4, 3, 4),
        (4096, 4096, 2048, 128, 128): (4, 1, 4),
        (4096, 4096, 4096, 16, 16): (2, 3, 1),
        (4096, 4096, 4096, 32, 32): (4, 4, 2),
        (4096, 4096, 4096, 64, 64): (1, 3, 2),
        (4096, 4096, 4096, 128, 128): (4, 1, 4),
        (4096, 4096, 8192, 16, 16): (4, 2, 1),
        (4096, 4096, 8192, 32, 32): (4, 3, 2),
        (4096, 4096, 8192, 64, 64): (4, 3, 2),
        (4096, 4096, 8192, 128, 128): (4, 1, 4),
        (4096, 4096, 16384, 16, 16): (1, 1, 1),
        (4096, 4096, 16384, 32, 32): (4, 4, 1),
        (4096, 4096, 16384, 64, 64): (4, 3, 2),
        (4096, 4096, 16384, 128, 128): (4, 1, 4),
        (4096, 4096, 32768, 16, 16): (4, 2, 1),
        (4096, 4096, 32768, 32, 32): (5, 3, 1),
        (4096, 4096, 32768, 64, 64): (5, 3, 2),
        (4096, 4096, 32768, 128, 128): (4, 1, 4),
        (4096, 4096, 65536, 16, 16): (4, 2, 1),
        (4096, 4096, 65536, 32, 32): (2, 3, 1),
        (4096, 4096, 65536, 64, 64): (2, 3, 2),
        (4096, 4096, 65536, 128, 128): (4, 1, 4),
        (4096, 4096, 131072, 16, 16): (1, 1, 4),
        (4096, 4096, 131072, 32, 32): (4, 2, 1),
        (4096, 4096, 131072, 64, 64): (5, 3, 2),
        (4096, 4096, 131072, 128, 128): (4, 1, 4),
        (8192, 8192, 256, 16, 16): (4, 4, 1),
        (8192, 8192, 256, 32, 32): (4, 3, 4),
        (8192, 8192, 256, 64, 64): (4, 3, 4),
        (8192, 8192, 256, 128, 128): (5, 1, 4),
        (8192, 8192, 512, 16, 16): (4, 5, 1),
        (8192, 8192, 512, 32, 32): (4, 4, 2),
        (8192, 8192, 512, 64, 64): (4, 4, 4),
        (8192, 8192, 512, 128, 128): (4, 2, 8),
        (8192, 8192, 1024, 16, 16): (4, 5, 1),
        (8192, 8192, 1024, 32, 32): (4, 4, 2),
        (8192, 8192, 1024, 64, 64): (4, 4, 4),
        (8192, 8192, 1024, 128, 128): (4, 1, 4),
        (8192, 8192, 2048, 16, 16): (4, 3, 1),
        (8192, 8192, 2048, 32, 32): (4, 4, 1),
        (8192, 8192, 2048, 64, 64): (4, 3, 4),
        (8192, 8192, 2048, 128, 128): (4, 1, 4),
        (8192, 8192, 4096, 16, 16): (4, 3, 1),
        (8192, 8192, 4096, 32, 32): (4, 4, 1),
        (8192, 8192, 4096, 64, 64): (4, 3, 2),
        (8192, 8192, 4096, 128, 128): (4, 1, 4),
        (8192, 8192, 8192, 16, 16): (4, 2, 1),
        (8192, 8192, 8192, 32, 32): (4, 4, 1),
        (8192, 8192, 8192, 64, 64): (4, 3, 2),
        (8192, 8192, 8192, 128, 128): (4, 1, 4),
        (8192, 8192, 16384, 16, 16): (4, 2, 1),
        (8192, 8192, 16384, 32, 32): (4, 4, 1),
        (8192, 8192, 16384, 64, 64): (4, 3, 2),
        (8192, 8192, 16384, 128, 128): (4, 1, 4),
        (8192, 8192, 32768, 16, 16): (4, 2, 1),
        (8192, 8192, 32768, 32, 32): (4, 4, 1),
        (8192, 8192, 32768, 64, 64): (4, 4, 2),
        (8192, 8192, 32768, 128, 128): (4, 1, 4),
        (8192, 8192, 65536, 16, 16): (4, 2, 1),
        (8192, 8192, 65536, 32, 32): (4, 3, 1),
        (8192, 8192, 65536, 64, 64): (4, 3, 2),
        (8192, 8192, 65536, 128, 128): (4, 1, 4),
        (8192, 8192, 131072, 16, 16): (4, 1, 4),
        (8192, 8192, 131072, 32, 32): (4, 2, 1),
        (8192, 8192, 131072, 64, 64): (4, 3, 2),
        (8192, 8192, 131072, 128, 128): (4, 1, 4),
        (16384, 16384, 256, 16, 16): (4, 4, 1),
        (16384, 16384, 256, 32, 32): (4, 4, 2),
        (16384, 16384, 256, 64, 64): (4, 4, 4),
        (16384, 16384, 256, 128, 128): (4, 2, 8),
        (16384, 16384, 512, 16, 16): (4, 3, 1),
        (16384, 16384, 512, 32, 32): (4, 5, 2),
        (16384, 16384, 512, 64, 64): (4, 4, 4),
        (16384, 16384, 512, 128, 128): (4, 2, 8),
        (16384, 16384, 1024, 16, 16): (4, 3, 1),
        (16384, 16384, 1024, 32, 32): (4, 4, 1),
        (16384, 16384, 1024, 64, 64): (4, 4, 4),
        (16384, 16384, 1024, 128, 128): (4, 1, 4),
        (16384, 16384, 2048, 16, 16): (4, 3, 1),
        (16384, 16384, 2048, 32, 32): (4, 4, 1),
        (16384, 16384, 2048, 64, 64): (4, 4, 4),
        (16384, 16384, 2048, 128, 128): (4, 1, 4),
        (16384, 16384, 4096, 16, 16): (4, 2, 1),
        (16384, 16384, 4096, 32, 32): (4, 4, 1),
        (16384, 16384, 4096, 64, 64): (4, 3, 2),
        (16384, 16384, 4096, 128, 128): (4, 1, 4),
        (16384, 16384, 8192, 16, 16): (4, 2, 1),
        (16384, 16384, 8192, 32, 32): (4, 4, 1),
        (16384, 16384, 8192, 64, 64): (4, 3, 2),
        (16384, 16384, 8192, 128, 128): (4, 1, 4),
        (16384, 16384, 16384, 16, 16): (4, 2, 1),
        (16384, 16384, 16384, 32, 32): (4, 5, 1),
        (16384, 16384, 16384, 64, 64): (4, 3, 2),
        (16384, 16384, 16384, 128, 128): (4, 1, 4),
        (16384, 16384, 32768, 16, 16): (4, 2, 1),
        (16384, 16384, 32768, 32, 32): (4, 6, 1),
        (16384, 16384, 32768, 64, 64): (4, 3, 2),
        (16384, 16384, 32768, 128, 128): (4, 1, 4),
        (16384, 16384, 65536, 16, 16): (4, 2, 1),
        (16384, 16384, 65536, 32, 32): (4, 2, 1),
        (16384, 16384, 65536, 64, 64): (4, 3, 2),
        (16384, 16384, 65536, 128, 128): (4, 1, 4),
        (16384, 16384, 131072, 16, 16): (4, 1, 4),
        (16384, 16384, 131072, 32, 32): (4, 2, 1),
        (16384, 16384, 131072, 64, 64): (4, 3, 2),
        (16384, 16384, 131072, 128, 128): (4, 1, 4),
    },
    ("bsr_dense_mm", "NVIDIA A100-SXM4-80GB", (0, torch.float16, 0.7)): {
        (256, 256, 256, 16, 16): (5, 1, 4),
        (256, 256, 256, 32, 32): (4, 3, 4),
        (256, 256, 256, 64, 64): (3, 3, 4),
        (256, 256, 256, 128, 128): (4, 2, 8),
        (256, 256, 512, 16, 16): (3, 1, 4),
        (256, 256, 512, 32, 32): (4, 1, 4),
        (256, 256, 512, 64, 64): (4, 3, 4),
        (256, 256, 512, 128, 128): (4, 2, 8),
        (256, 256, 1024, 16, 16): (4, 1, 4),
        (256, 256, 1024, 32, 32): (4, 3, 4),
        (256, 256, 1024, 64, 64): (4, 3, 4),
        (256, 256, 1024, 128, 128): (4, 2, 8),
        (256, 256, 2048, 16, 16): (5, 1, 2),
        (256, 256, 2048, 32, 32): (5, 2, 4),
        (256, 256, 2048, 64, 64): (4, 3, 4),
        (256, 256, 2048, 128, 128): (5, 2, 8),
        (256, 256, 4096, 16, 16): (4, 1, 1),
        (256, 256, 4096, 32, 32): (4, 2, 2),
        (256, 256, 4096, 64, 64): (5, 3, 4),
        (256, 256, 4096, 128, 128): (4, 2, 8),
        (256, 256, 8192, 16, 16): (4, 2, 1),
        (256, 256, 8192, 32, 32): (4, 2, 2),
        (256, 256, 8192, 64, 64): (4, 2, 4),
        (256, 256, 8192, 128, 128): (4, 2, 8),
        (256, 256, 16384, 16, 16): (5, 2, 1),
        (256, 256, 16384, 32, 32): (4, 2, 2),
        (256, 256, 16384, 64, 64): (3, 1, 4),
        (256, 256, 16384, 128, 128): (4, 1, 4),
        (256, 256, 32768, 16, 16): (3, 2, 1),
        (256, 256, 32768, 32, 32): (4, 3, 2),
        (256, 256, 32768, 64, 64): (2, 1, 4),
        (256, 256, 32768, 128, 128): (4, 1, 4),
        (256, 256, 65536, 16, 16): (4, 2, 1),
        (256, 256, 65536, 32, 32): (4, 3, 1),
        (256, 256, 65536, 64, 64): (5, 1, 2),
        (256, 256, 65536, 128, 128): (4, 1, 4),
        (256, 256, 131072, 16, 16): (4, 1, 2),
        (256, 256, 131072, 32, 32): (4, 2, 1),
        (256, 256, 131072, 64, 64): (3, 1, 2),
        (256, 256, 131072, 128, 128): (4, 1, 4),
        (512, 512, 256, 16, 16): (4, 1, 4),
        (512, 512, 256, 32, 32): (3, 1, 4),
        (512, 512, 256, 64, 64): (4, 3, 4),
        (512, 512, 256, 128, 128): (3, 2, 8),
        (512, 512, 512, 16, 16): (4, 1, 4),
        (512, 512, 512, 32, 32): (4, 1, 4),
        (512, 512, 512, 64, 64): (5, 3, 4),
        (512, 512, 512, 128, 128): (4, 2, 8),
        (512, 512, 1024, 16, 16): (4, 3, 1),
        (512, 512, 1024, 32, 32): (4, 1, 4),
        (512, 512, 1024, 64, 64): (4, 5, 4),
        (512, 512, 1024, 128, 128): (4, 2, 8),
        (512, 512, 2048, 16, 16): (3, 3, 1),
        (512, 512, 2048, 32, 32): (4, 3, 1),
        (512, 512, 2048, 64, 64): (3, 3, 2),
        (512, 512, 2048, 128, 128): (4, 2, 8),
        (512, 512, 4096, 16, 16): (4, 2, 1),
        (512, 512, 4096, 32, 32): (4, 4, 2),
        (512, 512, 4096, 64, 64): (4, 3, 4),
        (512, 512, 4096, 128, 128): (4, 2, 8),
        (512, 512, 8192, 16, 16): (4, 2, 1),
        (512, 512, 8192, 32, 32): (5, 3, 1),
        (512, 512, 8192, 64, 64): (5, 2, 4),
        (512, 512, 8192, 128, 128): (3, 1, 4),
        (512, 512, 16384, 16, 16): (4, 2, 1),
        (512, 512, 16384, 32, 32): (4, 3, 1),
        (512, 512, 16384, 64, 64): (5, 3, 4),
        (512, 512, 16384, 128, 128): (3, 1, 4),
        (512, 512, 32768, 16, 16): (5, 3, 1),
        (512, 512, 32768, 32, 32): (5, 3, 1),
        (512, 512, 32768, 64, 64): (5, 3, 2),
        (512, 512, 32768, 128, 128): (5, 1, 4),
        (512, 512, 65536, 16, 16): (4, 3, 1),
        (512, 512, 65536, 32, 32): (4, 3, 1),
        (512, 512, 65536, 64, 64): (4, 3, 2),
        (512, 512, 65536, 128, 128): (5, 1, 4),
        (512, 512, 131072, 16, 16): (4, 2, 2),
        (512, 512, 131072, 32, 32): (4, 2, 1),
        (512, 512, 131072, 64, 64): (4, 2, 4),
        (512, 512, 131072, 128, 128): (3, 1, 4),
        (1024, 1024, 256, 16, 16): (4, 4, 1),
        (1024, 1024, 256, 32, 32): (4, 1, 4),
        (1024, 1024, 256, 64, 64): (4, 3, 4),
        (1024, 1024, 256, 128, 128): (4, 2, 8),
        (1024, 1024, 512, 16, 16): (4, 3, 1),
        (1024, 1024, 512, 32, 32): (5, 4, 2),
        (1024, 1024, 512, 64, 64): (4, 4, 4),
        (1024, 1024, 512, 128, 128): (3, 2, 8),
        (1024, 1024, 1024, 16, 16): (4, 3, 1),
        (1024, 1024, 1024, 32, 32): (1, 3, 1),
        (1024, 1024, 1024, 64, 64): (1, 3, 2),
        (1024, 1024, 1024, 128, 128): (1, 2, 8),
        (1024, 1024, 2048, 16, 16): (3, 3, 1),
        (1024, 1024, 2048, 32, 32): (4, 4, 1),
        (1024, 1024, 2048, 64, 64): (3, 3, 2),
        (1024, 1024, 2048, 128, 128): (4, 2, 8),
        (1024, 1024, 4096, 16, 16): (4, 3, 1),
        (1024, 1024, 4096, 32, 32): (4, 3, 1),
        (1024, 1024, 4096, 64, 64): (4, 3, 4),
        (1024, 1024, 4096, 128, 128): (4, 2, 8),
        (1024, 1024, 8192, 16, 16): (4, 3, 1),
        (1024, 1024, 8192, 32, 32): (4, 3, 1),
        (1024, 1024, 8192, 64, 64): (4, 3, 2),
        (1024, 1024, 8192, 128, 128): (4, 1, 4),
        (1024, 1024, 16384, 16, 16): (4, 3, 1),
        (1024, 1024, 16384, 32, 32): (4, 3, 1),
        (1024, 1024, 16384, 64, 64): (4, 3, 2),
        (1024, 1024, 16384, 128, 128): (4, 1, 4),
        (1024, 1024, 32768, 16, 16): (4, 3, 1),
        (1024, 1024, 32768, 32, 32): (5, 3, 1),
        (1024, 1024, 32768, 64, 64): (5, 3, 2),
        (1024, 1024, 32768, 128, 128): (4, 1, 4),
        (1024, 1024, 65536, 16, 16): (4, 3, 1),
        (1024, 1024, 65536, 32, 32): (5, 3, 1),
        (1024, 1024, 65536, 64, 64): (5, 3, 2),
        (1024, 1024, 65536, 128, 128): (4, 1, 4),
        (1024, 1024, 131072, 16, 16): (1, 1, 4),
        (1024, 1024, 131072, 32, 32): (4, 2, 1),
        (1024, 1024, 131072, 64, 64): (4, 3, 2),
        (1024, 1024, 131072, 128, 128): (4, 1, 4),
        (2048, 2048, 256, 16, 16): (4, 4, 1),
        (2048, 2048, 256, 32, 32): (4, 4, 2),
        (2048, 2048, 256, 64, 64): (5, 3, 4),
        (2048, 2048, 256, 128, 128): (4, 2, 8),
        (2048, 2048, 512, 16, 16): (3, 3, 1),
        (2048, 2048, 512, 32, 32): (6, 3, 2),
        (2048, 2048, 512, 64, 64): (6, 3, 4),
        (2048, 2048, 512, 128, 128): (4, 2, 8),
        (2048, 2048, 1024, 16, 16): (5, 3, 1),
        (2048, 2048, 1024, 32, 32): (4, 4, 2),
        (2048, 2048, 1024, 64, 64): (5, 3, 4),
        (2048, 2048, 1024, 128, 128): (4, 2, 8),
        (2048, 2048, 2048, 16, 16): (2, 3, 1),
        (2048, 2048, 2048, 32, 32): (2, 4, 1),
        (2048, 2048, 2048, 64, 64): (2, 3, 2),
        (2048, 2048, 2048, 128, 128): (4, 2, 8),
        (2048, 2048, 4096, 16, 16): (2, 3, 1),
        (2048, 2048, 4096, 32, 32): (4, 3, 2),
        (2048, 2048, 4096, 64, 64): (4, 3, 2),
        (2048, 2048, 4096, 128, 128): (4, 1, 4),
        (2048, 2048, 8192, 16, 16): (4, 2, 1),
        (2048, 2048, 8192, 32, 32): (4, 3, 2),
        (2048, 2048, 8192, 64, 64): (4, 3, 2),
        (2048, 2048, 8192, 128, 128): (4, 1, 4),
        (2048, 2048, 16384, 16, 16): (4, 2, 1),
        (2048, 2048, 16384, 32, 32): (4, 4, 2),
        (2048, 2048, 16384, 64, 64): (4, 3, 2),
        (2048, 2048, 16384, 128, 128): (4, 1, 4),
        (2048, 2048, 32768, 16, 16): (4, 2, 1),
        (2048, 2048, 32768, 32, 32): (5, 4, 2),
        (2048, 2048, 32768, 64, 64): (5, 3, 2),
        (2048, 2048, 32768, 128, 128): (4, 1, 4),
        (2048, 2048, 65536, 16, 16): (4, 2, 1),
        (2048, 2048, 65536, 32, 32): (5, 4, 1),
        (2048, 2048, 65536, 64, 64): (5, 3, 2),
        (2048, 2048, 65536, 128, 128): (4, 1, 4),
        (2048, 2048, 131072, 16, 16): (4, 1, 4),
        (2048, 2048, 131072, 32, 32): (4, 2, 1),
        (2048, 2048, 131072, 64, 64): (4, 3, 2),
        (2048, 2048, 131072, 128, 128): (4, 1, 4),
        (4096, 4096, 256, 16, 16): (4, 4, 1),
        (4096, 4096, 256, 32, 32): (4, 3, 2),
        (4096, 4096, 256, 64, 64): (3, 3, 4),
        (4096, 4096, 256, 128, 128): (3, 2, 8),
        (4096, 4096, 512, 16, 16): (4, 4, 1),
        (4096, 4096, 512, 32, 32): (4, 5, 2),
        (4096, 4096, 512, 64, 64): (6, 3, 4),
        (4096, 4096, 512, 128, 128): (6, 2, 8),
        (4096, 4096, 1024, 16, 16): (4, 3, 1),
        (4096, 4096, 1024, 32, 32): (4, 4, 2),
        (4096, 4096, 1024, 64, 64): (4, 3, 4),
        (4096, 4096, 1024, 128, 128): (4, 2, 8),
        (4096, 4096, 2048, 16, 16): (4, 3, 1),
        (4096, 4096, 2048, 32, 32): (3, 4, 2),
        (4096, 4096, 2048, 64, 64): (4, 3, 4),
        (4096, 4096, 2048, 128, 128): (4, 1, 4),
        (4096, 4096, 4096, 16, 16): (4, 2, 1),
        (4096, 4096, 4096, 32, 32): (5, 4, 2),
        (4096, 4096, 4096, 64, 64): (1, 3, 2),
        (4096, 4096, 4096, 128, 128): (2, 1, 4),
        (4096, 4096, 8192, 16, 16): (2, 2, 1),
        (4096, 4096, 8192, 32, 32): (4, 4, 2),
        (4096, 4096, 8192, 64, 64): (4, 3, 2),
        (4096, 4096, 8192, 128, 128): (2, 1, 4),
        (4096, 4096, 16384, 16, 16): (2, 3, 1),
        (4096, 4096, 16384, 32, 32): (4, 4, 1),
        (4096, 4096, 16384, 64, 64): (4, 3, 2),
        (4096, 4096, 16384, 128, 128): (3, 1, 4),
        (4096, 4096, 32768, 16, 16): (6, 2, 1),
        (4096, 4096, 32768, 32, 32): (5, 4, 1),
        (4096, 4096, 32768, 64, 64): (5, 3, 2),
        (4096, 4096, 32768, 128, 128): (2, 1, 4),
        (4096, 4096, 65536, 16, 16): (6, 2, 1),
        (4096, 4096, 65536, 32, 32): (2, 3, 1),
        (4096, 4096, 65536, 64, 64): (2, 3, 2),
        (4096, 4096, 65536, 128, 128): (1, 1, 4),
        (4096, 4096, 131072, 16, 16): (1, 1, 4),
        (4096, 4096, 131072, 32, 32): (4, 2, 1),
        (4096, 4096, 131072, 64, 64): (7, 3, 2),
        (4096, 4096, 131072, 128, 128): (1, 1, 4),
        (8192, 8192, 256, 16, 16): (4, 4, 1),
        (8192, 8192, 256, 32, 32): (4, 3, 4),
        (8192, 8192, 256, 64, 64): (4, 3, 4),
        (8192, 8192, 256, 128, 128): (5, 2, 8),
        (8192, 8192, 512, 16, 16): (4, 5, 1),
        (8192, 8192, 512, 32, 32): (4, 4, 2),
        (8192, 8192, 512, 64, 64): (4, 4, 4),
        (8192, 8192, 512, 128, 128): (4, 2, 8),
        (8192, 8192, 1024, 16, 16): (4, 5, 1),
        (8192, 8192, 1024, 32, 32): (4, 4, 2),
        (8192, 8192, 1024, 64, 64): (4, 4, 4),
        (8192, 8192, 1024, 128, 128): (4, 1, 4),
        (8192, 8192, 2048, 16, 16): (4, 3, 1),
        (8192, 8192, 2048, 32, 32): (4, 4, 2),
        (8192, 8192, 2048, 64, 64): (4, 3, 2),
        (8192, 8192, 2048, 128, 128): (4, 1, 4),
        (8192, 8192, 4096, 16, 16): (4, 3, 1),
        (8192, 8192, 4096, 32, 32): (4, 4, 1),
        (8192, 8192, 4096, 64, 64): (4, 3, 2),
        (8192, 8192, 4096, 128, 128): (4, 1, 4),
        (8192, 8192, 8192, 16, 16): (4, 2, 1),
        (8192, 8192, 8192, 32, 32): (4, 4, 1),
        (8192, 8192, 8192, 64, 64): (4, 3, 2),
        (8192, 8192, 8192, 128, 128): (4, 1, 4),
        (8192, 8192, 16384, 16, 16): (4, 2, 1),
        (8192, 8192, 16384, 32, 32): (4, 4, 1),
        (8192, 8192, 16384, 64, 64): (4, 3, 2),
        (8192, 8192, 16384, 128, 128): (4, 1, 4),
        (8192, 8192, 32768, 16, 16): (4, 2, 1),
        (8192, 8192, 32768, 32, 32): (4, 4, 1),
        (8192, 8192, 32768, 64, 64): (4, 3, 2),
        (8192, 8192, 32768, 128, 128): (4, 1, 4),
        (8192, 8192, 65536, 16, 16): (4, 2, 1),
        (8192, 8192, 65536, 32, 32): (4, 3, 1),
        (8192, 8192, 65536, 64, 64): (4, 3, 2),
        (8192, 8192, 65536, 128, 128): (4, 1, 4),
        (8192, 8192, 131072, 16, 16): (4, 1, 4),
        (8192, 8192, 131072, 32, 32): (4, 2, 1),
        (8192, 8192, 131072, 64, 64): (4, 3, 2),
        (8192, 8192, 131072, 128, 128): (4, 1, 4),
        (16384, 16384, 256, 16, 16): (2, 7, 1),
        (16384, 16384, 256, 32, 32): (4, 4, 2),
        (16384, 16384, 256, 64, 64): (4, 4, 4),
        (16384, 16384, 256, 128, 128): (4, 2, 8),
        (16384, 16384, 512, 16, 16): (4, 3, 1),
        (16384, 16384, 512, 32, 32): (4, 5, 2),
        (16384, 16384, 512, 64, 64): (4, 4, 4),
        (16384, 16384, 512, 128, 128): (4, 2, 8),
        (16384, 16384, 1024, 16, 16): (4, 3, 1),
        (16384, 16384, 1024, 32, 32): (4, 4, 1),
        (16384, 16384, 1024, 64, 64): (4, 3, 2),
        (16384, 16384, 1024, 128, 128): (4, 1, 4),
        (16384, 16384, 2048, 16, 16): (4, 3, 1),
        (16384, 16384, 2048, 32, 32): (4, 4, 1),
        (16384, 16384, 2048, 64, 64): (4, 4, 4),
        (16384, 16384, 2048, 128, 128): (4, 1, 4),
        (16384, 16384, 4096, 16, 16): (4, 3, 1),
        (16384, 16384, 4096, 32, 32): (4, 4, 1),
        (16384, 16384, 4096, 64, 64): (4, 3, 2),
        (16384, 16384, 4096, 128, 128): (4, 1, 4),
        (16384, 16384, 8192, 16, 16): (2, 2, 1),
        (16384, 16384, 8192, 32, 32): (4, 4, 1),
        (16384, 16384, 8192, 64, 64): (4, 3, 2),
        (16384, 16384, 8192, 128, 128): (4, 1, 4),
        (16384, 16384, 16384, 16, 16): (4, 2, 1),
        (16384, 16384, 16384, 32, 32): (4, 4, 1),
        (16384, 16384, 16384, 64, 64): (4, 3, 2),
        (16384, 16384, 16384, 128, 128): (4, 1, 4),
        (16384, 16384, 32768, 16, 16): (2, 2, 1),
        (16384, 16384, 32768, 32, 32): (4, 5, 1),
        (16384, 16384, 32768, 64, 64): (4, 3, 2),
        (16384, 16384, 32768, 128, 128): (4, 1, 4),
        (16384, 16384, 65536, 16, 16): (4, 2, 1),
        (16384, 16384, 65536, 32, 32): (4, 4, 1),
        (16384, 16384, 65536, 64, 64): (4, 3, 2),
        (16384, 16384, 65536, 128, 128): (4, 1, 4),
        (16384, 16384, 131072, 16, 16): (4, 1, 4),
        (16384, 16384, 131072, 32, 32): (4, 2, 1),
        (16384, 16384, 131072, 64, 64): (4, 3, 2),
        (16384, 16384, 131072, 128, 128): (4, 1, 4),
    },
    ("scatter_mm", "NVIDIA A100-SXM4-80GB", (0, torch.float16, 0.5)): {
        (256, 256, 256, 16, 16): (5, 4, 16, 16, 1, 4),
        (256, 256, 256, 32, 32): (5, 2, 32, 16, 1, 4),
        (256, 256, 256, 64, 64): (4, 1, 32, 32, 1, 8),
        (256, 256, 256, 128, 128): (2, 1, 32, 32, 1, 4),
        (256, 256, 512, 16, 16): (4, 8, 16, 32, 1, 4),
        (256, 256, 512, 32, 32): (4, 8, 32, 32, 1, 8),
        (256, 256, 512, 64, 64): (4, 8, 32, 64, 1, 4),
        (256, 256, 512, 128, 128): (4, 8, 32, 64, 1, 4),
        (256, 256, 1024, 16, 16): (4, 2, 16, 64, 1, 2),
        (256, 256, 1024, 32, 32): (4, 16, 32, 64, 1, 2),
        (256, 256, 1024, 64, 64): (4, 16, 32, 64, 1, 4),
        (256, 256, 1024, 128, 128): (4, 16, 64, 64, 1, 8),
        (256, 256, 2048, 16, 16): (4, 16, 16, 64, 1, 1),
        (256, 256, 2048, 32, 32): (4, 16, 32, 64, 1, 2),
        (256, 256, 2048, 64, 64): (4, 16, 32, 64, 1, 4),
        (256, 256, 2048, 128, 128): (4, 16, 64, 64, 1, 4),
        (256, 256, 4096, 16, 16): (4, 32, 16, 64, 1, 1),
        (256, 256, 4096, 32, 32): (4, 32, 32, 64, 1, 2),
        (256, 256, 4096, 64, 64): (4, 64, 64, 64, 1, 4),
        (256, 256, 4096, 128, 128): (4, 32, 64, 64, 1, 4),
        (256, 256, 8192, 16, 16): (4, 64, 16, 64, 1, 1),
        (256, 256, 8192, 32, 32): (4, 128, 32, 64, 1, 2),
        (256, 256, 8192, 64, 64): (4, 64, 64, 64, 1, 4),
        (256, 256, 8192, 128, 128): (4, 64, 64, 64, 1, 4),
        (256, 256, 16384, 16, 16): (4, 128, 16, 64, 1, 1),
        (256, 256, 16384, 32, 32): (4, 16, 32, 64, 1, 2),
        (256, 256, 16384, 64, 64): (4, 32, 32, 128, 1, 4),
        (256, 256, 16384, 128, 128): (4, 16, 64, 64, 1, 4),
        (256, 256, 32768, 16, 16): (4, 64, 16, 64, 1, 1),
        (256, 256, 32768, 32, 32): (4, 32, 32, 64, 1, 2),
        (256, 256, 32768, 64, 64): (4, 32, 32, 128, 1, 4),
        (256, 256, 32768, 128, 128): (4, 32, 64, 64, 1, 4),
        (256, 256, 65536, 16, 16): (4, 128, 16, 64, 1, 1),
        (256, 256, 65536, 32, 32): (4, 16, 32, 64, 1, 2),
        (256, 256, 65536, 64, 64): (4, 16, 64, 64, 1, 2),
        (256, 256, 65536, 128, 128): (4, 32, 64, 64, 1, 4),
        (256, 256, 131072, 16, 16): (4, 64, 16, 64, 1, 1),
        (256, 256, 131072, 32, 32): (4, 2, 32, 64, 1, 2),
        (256, 256, 131072, 64, 64): (4, 32, 32, 128, 1, 4),
        (256, 256, 131072, 128, 128): (4, 32, 64, 64, 1, 4),
        (512, 512, 256, 16, 16): (4, 16, 16, 16, 1, 4),
        (512, 512, 256, 32, 32): (4, 16, 32, 16, 1, 4),
        (512, 512, 256, 64, 64): (4, 16, 64, 16, 1, 8),
        (512, 512, 256, 128, 128): (4, 16, 64, 16, 1, 4),
        (512, 512, 512, 16, 16): (2, 1, 16, 64, 1, 2),
        (512, 512, 512, 32, 32): (2, 4, 16, 32, 1, 1),
        (512, 512, 512, 64, 64): (2, 1, 32, 32, 1, 2),
        (512, 512, 512, 128, 128): (4, 8, 32, 64, 1, 4),
        (512, 512, 1024, 16, 16): (4, 8, 16, 64, 1, 1),
        (512, 512, 1024, 32, 32): (4, 16, 32, 64, 1, 2),
        (512, 512, 1024, 64, 64): (4, 16, 64, 64, 1, 4),
        (512, 512, 1024, 128, 128): (4, 16, 64, 64, 1, 4),
        (512, 512, 2048, 16, 16): (4, 16, 16, 64, 1, 4),
        (512, 512, 2048, 32, 32): (4, 16, 32, 64, 1, 2),
        (512, 512, 2048, 64, 64): (4, 16, 64, 64, 1, 8),
        (512, 512, 2048, 128, 128): (4, 16, 64, 64, 1, 4),
        (512, 512, 4096, 16, 16): (4, 32, 16, 128, 1, 2),
        (512, 512, 4096, 32, 32): (4, 32, 32, 64, 1, 2),
        (512, 512, 4096, 64, 64): (4, 32, 64, 64, 1, 4),
        (512, 512, 4096, 128, 128): (4, 32, 64, 64, 1, 4),
        (512, 512, 8192, 16, 16): (3, 16, 16, 128, 1, 2),
        (512, 512, 8192, 32, 32): (4, 64, 32, 64, 1, 2),
        (512, 512, 8192, 64, 64): (4, 128, 64, 64, 1, 2),
        (512, 512, 8192, 128, 128): (4, 64, 64, 64, 1, 4),
        (512, 512, 16384, 16, 16): (4, 32, 16, 64, 1, 1),
        (512, 512, 16384, 32, 32): (4, 64, 32, 64, 1, 2),
        (512, 512, 16384, 64, 64): (4, 16, 64, 64, 1, 4),
        (512, 512, 16384, 128, 128): (4, 32, 64, 64, 1, 4),
        (512, 512, 32768, 16, 16): (6, 16, 16, 128, 1, 2),
        (512, 512, 32768, 32, 32): (4, 64, 32, 64, 1, 2),
        (512, 512, 32768, 64, 64): (4, 32, 64, 64, 1, 2),
        (512, 512, 32768, 128, 128): (4, 16, 64, 64, 1, 4),
        (512, 512, 65536, 16, 16): (4, 32, 16, 64, 1, 1),
        (512, 512, 65536, 32, 32): (4, 64, 32, 64, 1, 2),
        (512, 512, 65536, 64, 64): (5, 32, 64, 64, 1, 2),
        (512, 512, 65536, 128, 128): (4, 16, 64, 64, 1, 4),
        (512, 512, 131072, 16, 16): (3, 32, 16, 128, 1, 2),
        (512, 512, 131072, 32, 32): (4, 64, 32, 64, 1, 2),
        (512, 512, 131072, 64, 64): (4, 32, 64, 64, 1, 2),
        (512, 512, 131072, 128, 128): (4, 16, 64, 64, 1, 4),
        (1024, 1024, 256, 16, 16): (4, 16, 16, 16, 1, 4),
        (1024, 1024, 256, 32, 32): (4, 16, 32, 16, 1, 4),
        (1024, 1024, 256, 64, 64): (4, 4, 64, 32, 1, 16),
        (1024, 1024, 256, 128, 128): (4, 16, 64, 16, 1, 8),
        (1024, 1024, 512, 16, 16): (4, 8, 16, 64, 1, 1),
        (1024, 1024, 512, 32, 32): (5, 8, 32, 64, 1, 2),
        (1024, 1024, 512, 64, 64): (4, 8, 32, 64, 1, 8),
        (1024, 1024, 512, 128, 128): (4, 8, 64, 64, 1, 8),
        (1024, 1024, 1024, 16, 16): (2, 2, 16, 64, 1, 2),
        (1024, 1024, 1024, 32, 32): (2, 8, 32, 64, 1, 2),
        (1024, 1024, 1024, 64, 64): (2, 8, 32, 128, 1, 4),
        (1024, 1024, 1024, 128, 128): (2, 8, 64, 64, 1, 4),
        (1024, 1024, 2048, 16, 16): (4, 16, 16, 128, 1, 2),
        (1024, 1024, 2048, 32, 32): (4, 32, 32, 64, 1, 2),
        (1024, 1024, 2048, 64, 64): (4, 16, 64, 64, 1, 4),
        (1024, 1024, 2048, 128, 128): (4, 32, 64, 64, 1, 4),
        (1024, 1024, 4096, 16, 16): (4, 16, 16, 128, 1, 2),
        (1024, 1024, 4096, 32, 32): (3, 32, 32, 64, 1, 2),
        (1024, 1024, 4096, 64, 64): (4, 32, 64, 64, 1, 4),
        (1024, 1024, 4096, 128, 128): (4, 32, 64, 64, 1, 4),
        (1024, 1024, 8192, 16, 16): (5, 16, 16, 128, 1, 2),
        (1024, 1024, 8192, 32, 32): (4, 32, 32, 64, 1, 2),
        (1024, 1024, 8192, 64, 64): (3, 64, 64, 64, 3, 2),
        (1024, 1024, 8192, 128, 128): (4, 32, 64, 64, 1, 4),
        (1024, 1024, 16384, 16, 16): (4, 16, 16, 128, 1, 2),
        (1024, 1024, 16384, 32, 32): (3, 32, 32, 64, 1, 2),
        (1024, 1024, 16384, 64, 64): (4, 16, 64, 64, 3, 2),
        (1024, 1024, 16384, 128, 128): (4, 32, 128, 64, 1, 4),
        (1024, 1024, 32768, 16, 16): (4, 16, 16, 128, 1, 2),
        (1024, 1024, 32768, 32, 32): (3, 32, 32, 64, 1, 2),
        (1024, 1024, 32768, 64, 64): (4, 16, 64, 64, 3, 2),
        (1024, 1024, 32768, 128, 128): (4, 8, 128, 64, 2, 4),
        (1024, 1024, 65536, 16, 16): (4, 8, 16, 128, 1, 2),
        (1024, 1024, 65536, 32, 32): (4, 16, 32, 64, 1, 2),
        (1024, 1024, 65536, 64, 64): (4, 16, 64, 64, 3, 2),
        (1024, 1024, 65536, 128, 128): (5, 8, 128, 64, 2, 4),
        (1024, 1024, 131072, 16, 16): (4, 8, 16, 128, 1, 2),
        (1024, 1024, 131072, 32, 32): (4, 16, 32, 64, 1, 2),
        (1024, 1024, 131072, 64, 64): (5, 16, 64, 64, 3, 2),
        (1024, 1024, 131072, 128, 128): (4, 8, 128, 64, 2, 4),
        (2048, 2048, 256, 16, 16): (4, 4, 16, 64, 1, 8),
        (2048, 2048, 256, 32, 32): (4, 8, 32, 32, 1, 8),
        (2048, 2048, 256, 64, 64): (4, 16, 64, 16, 1, 8),
        (2048, 2048, 256, 128, 128): (4, 4, 128, 32, 3, 8),
        (2048, 2048, 512, 16, 16): (4, 8, 16, 64, 1, 2),
        (2048, 2048, 512, 32, 32): (4, 4, 32, 64, 1, 2),
        (2048, 2048, 512, 64, 64): (4, 4, 64, 64, 1, 8),
        (2048, 2048, 512, 128, 128): (4, 8, 64, 64, 1, 4),
        (2048, 2048, 1024, 16, 16): (3, 8, 16, 64, 1, 2),
        (2048, 2048, 1024, 32, 32): (4, 16, 32, 64, 1, 2),
        (2048, 2048, 1024, 64, 64): (4, 8, 64, 64, 1, 4),
        (2048, 2048, 1024, 128, 128): (4, 8, 128, 64, 1, 4),
        (2048, 2048, 2048, 16, 16): (4, 4, 16, 128, 1, 2),
        (2048, 2048, 2048, 32, 32): (2, 16, 32, 64, 1, 2),
        (2048, 2048, 2048, 64, 64): (2, 8, 64, 64, 1, 4),
        (2048, 2048, 2048, 128, 128): (2, 8, 128, 64, 1, 4),
        (2048, 2048, 4096, 16, 16): (4, 2, 16, 128, 1, 2),
        (2048, 2048, 4096, 32, 32): (4, 16, 32, 64, 1, 2),
        (2048, 2048, 4096, 64, 64): (4, 32, 64, 64, 3, 2),
        (2048, 2048, 4096, 128, 128): (4, 8, 128, 64, 1, 4),
        (2048, 2048, 8192, 16, 16): (5, 4, 16, 128, 1, 2),
        (2048, 2048, 8192, 32, 32): (4, 64, 32, 64, 1, 2),
        (2048, 2048, 8192, 64, 64): (4, 8, 64, 64, 3, 2),
        (2048, 2048, 8192, 128, 128): (4, 8, 128, 64, 1, 4),
        (2048, 2048, 16384, 16, 16): (3, 2, 16, 128, 1, 2),
        (2048, 2048, 16384, 32, 32): (4, 8, 32, 64, 1, 2),
        (2048, 2048, 16384, 64, 64): (4, 8, 64, 64, 3, 2),
        (2048, 2048, 16384, 128, 128): (4, 4, 128, 64, 1, 4),
        (2048, 2048, 32768, 16, 16): (6, 2, 16, 128, 1, 2),
        (2048, 2048, 32768, 32, 32): (5, 8, 32, 64, 1, 2),
        (2048, 2048, 32768, 64, 64): (6, 4, 64, 64, 3, 2),
        (2048, 2048, 32768, 128, 128): (3, 4, 128, 64, 1, 4),
        (2048, 2048, 65536, 16, 16): (7, 2, 16, 128, 1, 2),
        (2048, 2048, 65536, 32, 32): (3, 1, 32, 128, 1, 2),
        (2048, 2048, 65536, 64, 64): (5, 4, 64, 64, 3, 2),
        (2048, 2048, 65536, 128, 128): (5, 1, 128, 64, 2, 4),
        (2048, 2048, 131072, 16, 16): (3, 2, 16, 128, 1, 2),
        (2048, 2048, 131072, 32, 32): (4, 2, 32, 128, 1, 4),
        (2048, 2048, 131072, 64, 64): (4, 1, 64, 64, 3, 2),
        (2048, 2048, 131072, 128, 128): (3, 1, 128, 64, 2, 4),
        (4096, 4096, 256, 16, 16): (5, 8, 16, 32, 1, 4),
        (4096, 4096, 256, 32, 32): (4, 16, 32, 16, 2, 4),
        (4096, 4096, 256, 64, 64): (4, 8, 64, 32, 1, 4),
        (4096, 4096, 256, 128, 128): (4, 4, 128, 32, 1, 4),
        (4096, 4096, 512, 16, 16): (4, 2, 16, 128, 1, 2),
        (4096, 4096, 512, 32, 32): (4, 8, 32, 64, 1, 2),
        (4096, 4096, 512, 64, 64): (4, 4, 64, 64, 1, 4),
        (4096, 4096, 512, 128, 128): (4, 8, 128, 64, 2, 4),
        (4096, 4096, 1024, 16, 16): (4, 8, 16, 128, 1, 2),
        (4096, 4096, 1024, 32, 32): (4, 8, 32, 64, 1, 2),
        (4096, 4096, 1024, 64, 64): (4, 16, 64, 64, 1, 4),
        (4096, 4096, 1024, 128, 128): (4, 16, 128, 64, 2, 4),
        (4096, 4096, 2048, 16, 16): (5, 8, 16, 128, 1, 2),
        (4096, 4096, 2048, 32, 32): (3, 4, 32, 64, 1, 2),
        (4096, 4096, 2048, 64, 64): (3, 16, 64, 64, 3, 2),
        (4096, 4096, 2048, 128, 128): (4, 32, 128, 64, 2, 4),
        (4096, 4096, 4096, 16, 16): (1, 2, 16, 128, 1, 2),
        (4096, 4096, 4096, 32, 32): (3, 4, 32, 64, 3, 2),
        (4096, 4096, 4096, 64, 64): (1, 1, 64, 64, 4, 4),
        (4096, 4096, 4096, 128, 128): (1, 1, 128, 128, 1, 8),
        (4096, 4096, 8192, 16, 16): (5, 8, 16, 128, 1, 2),
        (4096, 4096, 8192, 32, 32): (4, 4, 32, 64, 1, 2),
        (4096, 4096, 8192, 64, 64): (4, 16, 64, 64, 3, 2),
        (4096, 4096, 8192, 128, 128): (4, 16, 128, 64, 2, 4),
        (4096, 4096, 16384, 16, 16): (4, 8, 16, 128, 1, 2),
        (4096, 4096, 16384, 32, 32): (6, 2, 32, 64, 1, 2),
        (4096, 4096, 16384, 64, 64): (4, 16, 64, 64, 3, 2),
        (4096, 4096, 16384, 128, 128): (4, 16, 128, 64, 2, 4),
        (4096, 4096, 32768, 16, 16): (2, 8, 16, 128, 1, 2),
        (4096, 4096, 32768, 32, 32): (3, 1, 32, 128, 1, 4),
        (4096, 4096, 32768, 64, 64): (5, 8, 64, 64, 3, 2),
        (4096, 4096, 32768, 128, 128): (5, 16, 128, 64, 2, 4),
        (4096, 4096, 65536, 16, 16): (6, 8, 16, 128, 1, 2),
        (4096, 4096, 65536, 32, 32): (5, 1, 32, 128, 1, 4),
        (4096, 4096, 65536, 64, 64): (3, 8, 64, 64, 3, 2),
        (4096, 4096, 65536, 128, 128): (3, 16, 128, 64, 2, 4),
        (4096, 4096, 131072, 16, 16): (5, 8, 16, 128, 1, 2),
        (4096, 4096, 131072, 32, 32): (5, 4, 32, 64, 1, 2),
        (4096, 4096, 131072, 64, 64): (5, 8, 64, 64, 3, 2),
        (4096, 4096, 131072, 128, 128): (4, 16, 128, 64, 2, 4),
        (8192, 8192, 256, 16, 16): (4, 16, 16, 16, 1, 4),
        (8192, 8192, 256, 32, 32): (4, 16, 32, 16, 4, 4),
        (8192, 8192, 256, 64, 64): (4, 16, 64, 16, 3, 8),
        (8192, 8192, 256, 128, 128): (4, 16, 128, 16, 1, 2),
        (8192, 8192, 512, 16, 16): (5, 8, 16, 64, 1, 4),
        (8192, 8192, 512, 32, 32): (4, 4, 32, 64, 1, 2),
        (8192, 8192, 512, 64, 64): (4, 4, 64, 64, 1, 4),
        (8192, 8192, 512, 128, 128): (4, 8, 128, 64, 2, 4),
        (8192, 8192, 1024, 16, 16): (4, 16, 16, 64, 1, 8),
        (8192, 8192, 1024, 32, 32): (4, 4, 32, 64, 1, 2),
        (8192, 8192, 1024, 64, 64): (4, 16, 64, 64, 3, 2),
        (8192, 8192, 1024, 128, 128): (4, 16, 128, 64, 2, 4),
        (8192, 8192, 2048, 16, 16): (5, 2, 16, 128, 1, 2),
        (8192, 8192, 2048, 32, 32): (4, 16, 32, 64, 1, 2),
        (8192, 8192, 2048, 64, 64): (4, 16, 64, 64, 3, 2),
        (8192, 8192, 2048, 128, 128): (6, 16, 128, 64, 2, 4),
        (8192, 8192, 4096, 16, 16): (4, 2, 16, 128, 1, 2),
        (8192, 8192, 4096, 32, 32): (4, 4, 32, 64, 1, 2),
        (8192, 8192, 4096, 64, 64): (3, 16, 64, 64, 3, 2),
        (8192, 8192, 4096, 128, 128): (3, 64, 128, 64, 2, 4),
        (8192, 8192, 8192, 16, 16): (3, 2, 16, 128, 1, 2),
        (8192, 8192, 8192, 32, 32): (2, 4, 32, 128, 1, 4),
        (8192, 8192, 8192, 64, 64): (4, 4, 64, 64, 1, 4),
        (8192, 8192, 8192, 128, 128): (2, 2, 128, 128, 3, 8),
        (8192, 8192, 16384, 16, 16): (4, 8, 16, 128, 1, 2),
        (8192, 8192, 16384, 32, 32): (3, 4, 32, 64, 1, 2),
        (8192, 8192, 16384, 64, 64): (5, 8, 64, 64, 3, 2),
        (8192, 8192, 16384, 128, 128): (3, 16, 128, 64, 2, 4),
        (8192, 8192, 32768, 16, 16): (3, 2, 16, 128, 1, 2),
        (8192, 8192, 32768, 32, 32): (4, 4, 32, 64, 1, 2),
        (8192, 8192, 32768, 64, 64): (2, 8, 64, 64, 3, 2),
        (8192, 8192, 32768, 128, 128): (6, 16, 128, 64, 2, 4),
        (8192, 8192, 65536, 16, 16): (9, 2, 16, 128, 1, 2),
        (8192, 8192, 65536, 32, 32): (6, 4, 32, 64, 1, 2),
        (8192, 8192, 65536, 64, 64): (4, 8, 64, 64, 3, 2),
        (8192, 8192, 65536, 128, 128): (3, 16, 128, 64, 2, 4),
        (8192, 8192, 131072, 16, 16): (7, 2, 16, 128, 1, 2),
        (8192, 8192, 131072, 32, 32): (3, 8, 32, 64, 1, 2),
        (8192, 8192, 131072, 64, 64): (1, 8, 64, 64, 3, 2),
        (8192, 8192, 131072, 128, 128): (4, 16, 128, 64, 2, 4),
        (16384, 16384, 256, 16, 16): (5, 16, 16, 16, 1, 4),
        (16384, 16384, 256, 32, 32): (4, 16, 32, 16, 4, 4),
        (16384, 16384, 256, 64, 64): (4, 16, 64, 16, 3, 8),
        (16384, 16384, 256, 128, 128): (4, 16, 128, 16, 1, 2),
        (16384, 16384, 512, 16, 16): (4, 4, 16, 64, 1, 4),
        (16384, 16384, 512, 32, 32): (4, 4, 32, 64, 1, 2),
        (16384, 16384, 512, 64, 64): (4, 8, 64, 64, 1, 4),
        (16384, 16384, 512, 128, 128): (3, 8, 128, 64, 2, 4),
        (16384, 16384, 1024, 16, 16): (4, 2, 16, 128, 1, 2),
        (16384, 16384, 1024, 32, 32): (4, 8, 32, 64, 1, 2),
        (16384, 16384, 1024, 64, 64): (6, 16, 64, 64, 3, 2),
        (16384, 16384, 1024, 128, 128): (3, 16, 128, 64, 2, 4),
        (16384, 16384, 2048, 16, 16): (3, 2, 16, 128, 1, 2),
        (16384, 16384, 2048, 32, 32): (5, 8, 32, 64, 1, 2),
        (16384, 16384, 2048, 64, 64): (5, 16, 64, 64, 3, 2),
        (16384, 16384, 2048, 128, 128): (3, 32, 128, 64, 2, 4),
        (16384, 16384, 4096, 16, 16): (3, 2, 16, 128, 1, 2),
        (16384, 16384, 4096, 32, 32): (5, 4, 32, 64, 1, 2),
        (16384, 16384, 4096, 64, 64): (4, 16, 64, 64, 3, 2),
        (16384, 16384, 4096, 128, 128): (3, 16, 128, 64, 2, 4),
        (16384, 16384, 8192, 16, 16): (4, 2, 16, 128, 1, 2),
        (16384, 16384, 8192, 32, 32): (4, 4, 32, 64, 1, 2),
        (16384, 16384, 8192, 64, 64): (4, 8, 64, 64, 3, 2),
        (16384, 16384, 8192, 128, 128): (6, 32, 128, 64, 2, 4),
        (16384, 16384, 16384, 16, 16): (1, 2, 16, 256, 1, 4),
        (16384, 16384, 16384, 32, 32): (2, 4, 32, 128, 1, 4),
        (16384, 16384, 16384, 64, 64): (5, 4, 64, 64, 1, 4),
        (16384, 16384, 16384, 128, 128): (4, 8, 128, 64, 2, 4),
        (16384, 16384, 32768, 16, 16): (2, 2, 16, 128, 1, 2),
        (16384, 16384, 32768, 32, 32): (2, 4, 32, 64, 1, 2),
        (16384, 16384, 32768, 64, 64): (5, 4, 64, 64, 1, 4),
        (16384, 16384, 32768, 128, 128): (5, 8, 128, 64, 2, 4),
        (16384, 16384, 65536, 16, 16): (5, 2, 16, 128, 1, 2),
        (16384, 16384, 65536, 32, 32): (4, 2, 32, 64, 1, 2),
        (16384, 16384, 65536, 64, 64): (2, 4, 64, 64, 1, 4),
        (16384, 16384, 65536, 128, 128): (4, 8, 128, 64, 2, 4),
        (16384, 16384, 131072, 16, 16): (3, 2, 16, 128, 1, 2),
        (16384, 16384, 131072, 32, 32): (3, 4, 32, 64, 1, 2),
        (16384, 16384, 131072, 64, 64): (4, 4, 64, 64, 1, 4),
        (16384, 16384, 131072, 128, 128): (1, 8, 128, 64, 2, 4),
        (32768, 32768, 256, 16, 16): (4, 16, 16, 16, 1, 4),
        (32768, 32768, 512, 16, 16): (4, 2, 16, 128, 1, 2),
        (32768, 32768, 1024, 16, 16): (3, 2, 16, 128, 1, 2),
        (32768, 32768, 2048, 16, 16): (4, 2, 16, 128, 1, 2),
        (32768, 32768, 4096, 16, 16): (5, 4, 16, 64, 1, 1),
        (32768, 32768, 8192, 16, 16): (4, 4, 16, 64, 1, 1),
        (32768, 32768, 16384, 16, 16): (4, 4, 16, 64, 1, 1),
        (32768, 32768, 32768, 16, 16): (5, 4, 16, 64, 1, 1),
    },
    # END GENERATED DATA
}

if __name__ == "__main__":
    for op in ["scatter_mm", "bsr_dense_mm"]:
        main(op=op, force=False)
