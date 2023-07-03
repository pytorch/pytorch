import torch
import numpy as np
import argparse

from typing import Dict

# debug print
DEBUG_PRINT = False

################################################################################
# configuration for random tests setup
################################################################################
# maximum number of tensors as inputs
MAX_TENSOR = 6
# maximum tensor rank
MAX_TENSOR_DIM = 5
# maximum tensor size
MAX_TENSOR_SIZE = 2**20
# use a size 1 tensor for debug
DEBUG_TENSOR = False
# tensor device
DEVICE = "cuda"
# data type for tensors
DTYPE = torch.float
# factor sorta control the depth of the model
GRAPH_FACTOR = 2

################################################################################
# helper functions
################################################################################


class WrongResultException(Exception):
    pass

# randomly reduce tensor_shape while preserving it to be broadcast-compatible
# two thing are done here:
#   1. trim starting dimensions;
#   2. randomly clamp remaining dimension to be size 1;


def get_broadcast_compatible_shape(tensor_shape):
    max_dim = len(tensor_shape)
    num_b_dims = np.random.randint(0, max_dim + 1)
    trim_head = np.random.randint(0, min(num_b_dims + 1, max_dim))

    shape = tensor_shape[trim_head:max_dim]
    for i in np.random.choice(range(max_dim - trim_head),
                              num_b_dims - trim_head,
                              replace=False):
        shape[i] = 1
    return shape

# generate random topology using seed and also flags


def random_topology_test(seed, *inp_tensor_list):
    np.random.seed(int(seed.numpy().tolist()))
    tensor_list = [*inp_tensor_list]
    num_tensor = len(tensor_list)

    # randomly add available constant value
    num_const = np.random.randint(0, num_tensor + 1)
    const_list = np.random.random(num_const)

    if DEBUG_PRINT:
        for const_item in const_list:
            print("----- real number {:.10f}", const_item)

    # we require all tensor to be in a single dependency set
    def get_root(x, dependency_map):
        if x in dependency_map:
            return get_root(dependency_map[x], dependency_map)
        else:
            return x
    d_map: Dict[int, int] = {}
    num_sets = num_tensor
    candidate = list(range(num_tensor))

    unary_operations = [torch.sigmoid, torch.relu]
    binary_operations = [torch.add, torch.sub, torch.mul]
    u_op_size = len(unary_operations)
    b_op_size = len(binary_operations)

    num_operations = np.random.randint(num_sets - 1,
                                       num_sets * GRAPH_FACTOR)

    ret_list = []

    while num_operations >= 0 or num_sets > 1:
        # we start off with randomly pick a candidate and operation
        index = np.random.randint(0, len(candidate))
        op_index = np.random.randint(0, u_op_size + b_op_size)
        lh_index = candidate[index]
        rh_index = None
        out_tensor = None

        if DEBUG_PRINT:
            print("iteration {0}, num_sets{1}, candidates {2}, tensor_list {3}, lh_index {4}, op_index {5}".format(
                num_operations, num_sets, candidate, len(tensor_list), lh_index, op_index))
        if num_operations >= 0:
            num_operations -= 1
            if op_index < u_op_size:
                # unary operation, we just apply a random operation on candidate
                out_tensor = unary_operations[op_index](tensor_list[lh_index])
            else:
                # binary operation, we randomly choose the other operand:
                #   1. tensor on tensor operation -> rh_index
                #   2. tensor on const operation
                # we are not restricted to candidate tensor any more.
                op_2_index = np.random.randint(0, len(tensor_list) + num_const)

                if op_2_index < len(tensor_list):
                    if op_2_index == lh_index:
                        # if we are unlucky that we picked on the candidate again, just try
                        # another tensor
                        op_2_index = (op_2_index + 1) % len(tensor_list)
                    # [if rh_index: create binary operator output tensor]
                    rh_index = op_2_index
                else:
                    left = tensor_list[lh_index]
                    right = const_list[op_2_index - len(tensor_list)]
                    # if np.random.randint(0, 2) > 0:
                    #  left = const_list[op_2_index - len(tensor_list)]
                    #  right = tensor_list[lh_index]
                    out_tensor = binary_operations[op_index - u_op_size](left, right)
                if DEBUG_PRINT:
                    print("binary, op_2_index {0}, rh_index ?{1}".format(op_2_index, rh_index))
        else:
            # binary operation, we just randomly pick two candidates.
            # this is not the most efficient way to close dependency, as we could have
            # two candidate that are actually connected
            cand_index = np.random.randint(0, len(candidate))
            if cand_index == index:
                cand_index = (cand_index + 1) % len(candidate)
            # [if rh_index: create binary operator output tensor]
            rh_index = candidate[cand_index]
            if DEBUG_PRINT:
                print("binary rh_index ?{0}".format(rh_index))

        # update candidate should happen before we remove rh_index
        candidate[index] = len(tensor_list)
        lh_root = get_root(lh_index, d_map)
        # [if rh_index: create binary operator output tensor]
        if rh_index is not None:

            out_tensor = binary_operations[op_index - u_op_size](
                tensor_list[lh_index],
                tensor_list[rh_index])

            # remove rh_index from candidate if it is used
            if rh_index in candidate:
                # python remove(val), not by index
                candidate.remove(rh_index)

            # check if we join dependency sets:
            rh_root = get_root(rh_index, d_map)
            if lh_root != rh_root:
                num_sets -= 1
                # update dependency, no need to update d_map[rh_root] when
                # they are already pointing the same root
                d_map[rh_root] = len(tensor_list)

        # no joining, just update dependency
        d_map[lh_root] = len(tensor_list)

        # update candidate, this avoids us applying identical operation on
        # the same tensor(s)
        tensor_list.append(out_tensor)

    # TODO: we should mark
    #       union(random_sample(tensor_list[num_tensor:]), candidate) as outputs.
    #       which would ensure we have no dead branch and a connected computation
    #       graph. However, it won't work easily if we have broadcast.
    # I have disabled broadcast for now to focus on topology test.
    for ind in candidate:
        ret_list.append(tensor_list[ind])

    out_list = np.random.choice(
        range(num_tensor, len(tensor_list)),
        np.random.randint(0, len(tensor_list) - num_tensor),
        False)
    for ind in out_list:
        if ind not in candidate:
            ret_list.append(tensor_list[ind])

    if DEBUG_PRINT:
        print("ended with tensor_list: {0}".format(len(tensor_list)))

    return tuple(ret_list)


def prepareInputTensorsToRandomTopoTest(seed,
                                        max_tensor_num,
                                        max_tensor_dim,
                                        max_tensor_size,
                                        debug_tensor,
                                        device,
                                        dtype):
    # set seed to numpy as well as torch
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(0, seed))

    # seed to pass to torch.jit.trace
    seed_tensor = torch.tensor(np.random.randint(0, seed))

    # random number of input tensors
    num_tensor = np.random.randint(1, max_tensor_num)

    # prepare randomized tensor shape
    tensor_dim = np.random.randint(1, max_tensor_dim)
    tensor_shape = []
    numel = 1
    if debug_tensor:
        tensor_shape.append(1)
    else:
        for i in range(tensor_dim):
            size_i = np.random.randint(1, int(max_tensor_size / numel / (2**(tensor_dim - i))))
            size_i = min(size_i, 128 + size_i % 128)
            tensor_shape.insert(0, size_i)
            numel *= size_i

    if DEBUG_PRINT:
        print("output tensor shape: ", tensor_shape)

    # vvv BROADCASTING vvv
    # select tensors to be broadcasted
    # TODO: enable broadcasting when we fully support it.
    # num_broadcasted_tensors = np.random.randint(0, num_tensor)
    num_broadcasted_tensors = np.random.randint(0, 1)
    # we leave at least one tensor not broadcasted
    broadcasted_tensors_indices = np.random.choice(torch.arange(num_tensor),
                                                   num_broadcasted_tensors,
                                                   replace=False)

    # vvv PREPARING TENSORS vvv
    tensor_list = []
    for i in range(num_tensor):
        if i in broadcasted_tensors_indices:
            # get broadcast-compatible shape:
            # Note that we are not playing with stride here, as stride doesn't affect
            # codegen meaningfully.
            compatible_shape = get_broadcast_compatible_shape(tensor_shape)
            tensor_list.append(torch.randn(compatible_shape, device=device, dtype=dtype) * 100)
        else:
            tensor_list.append(torch.randn(tensor_shape, device=device, dtype=dtype) * 100)
    return seed_tensor, tensor_list


def reproString(current_seed, args):
    repro_str = "python {0}".format(__file__)
    if args.cuda_fuser:
        repro_str += " --cuda-fuser"
    if args.legacy_fuser:
        repro_str += " --legacy-fuser"
    if args.profiling_executor:
        repro_str += " --profiling-executor"
    if args.fp16:
        repro_str += " --fp16"
    if args.cpu:
        repro_str += " --cpu"
    repro_str += " --max-num-tensor {0} --max-tensor-dim {1} --max-tensor-size {2}"\
        " --depth-factor {3} --seed {4} --repro-run".format(
            args.max_num_tensor, args.max_tensor_dim, args.max_tensor_size,
            args.depth_factor, current_seed)
    return repro_str

################################################################################
# global seed to repro the test
################################################################################


def runDefaultTestWithSeed(seed):
    # prepare input tensors
    seed_tensor, tensor_list = prepareInputTensorsToRandomTopoTest(seed,
                                                                   MAX_TENSOR,
                                                                   MAX_TENSOR_DIM,
                                                                   MAX_TENSOR_SIZE,
                                                                   DEBUG_TENSOR,
                                                                   DEVICE,
                                                                   DTYPE)
    o = random_topology_test(seed_tensor, *tensor_list)
    traced_model = torch.jit.trace(random_topology_test, (seed_tensor, *tensor_list))
    jit_o = traced_model(seed_tensor, *tensor_list)  # possible profiling run
    jit_o = traced_model(seed_tensor, *tensor_list)
    validate_o = zip(o, jit_o)
    for oo, jit_oo in validate_o:
        if not oo.allclose(jit_oo, atol=1e-5, equal_nan=True):
            return False
    return True


def runTest(seed, args):
    # prepare input tensors
    seed_tensor, tensor_list = prepareInputTensorsToRandomTopoTest(seed,
                                                                   args.max_num_tensor,
                                                                   args.max_tensor_dim,
                                                                   args.max_tensor_size,
                                                                   args.debug_tensor,
                                                                   "cuda" if not args.cpu else "cpu",
                                                                   torch.float32 if not args.fp16 else torch.float16)

    # vvv run random generated topo test in eager vvv
    try:
        if DEBUG_PRINT:
            print("seed tensor: ", seed_tensor)
        o = random_topology_test(seed_tensor, *tensor_list)
        if DEBUG_PRINT:
            for out in o:
                print("val size: ", out.size())
    except Exception as err:
        raise Exception("Testing script failure with error message, repro by running:\n"
                        f"\t{reproString(seed, args)}") from err
    try:
        traced_model = torch.jit.trace(random_topology_test, (seed_tensor, *tensor_list))
        if DEBUG_PRINT:
            print("original graph: ", traced_model.graph)
        jit_o = traced_model(seed_tensor, *tensor_list)  # possible profiling run
        jit_o = traced_model(seed_tensor, *tensor_list)
        if DEBUG_PRINT:
            print("optimized graph: ", traced_model.graph_for(seed_tensor, *tensor_list))

        validate_o = zip(o, jit_o)
        for oo, jit_oo in validate_o:
            if not oo.allclose(jit_oo, equal_nan=True):
                print("eager output: ", oo)
                print("jit output: ", jit_oo)
                print("diff ", jit_oo - oo)
                raise WrongResultException()
    except WrongResultException as err:
        raise Exception("cuda fuser gives wrong results, repro by running:\n"
                        f"\t{reproString(seed, args)}") from err
    except Exception as err:
        raise Exception("something in cuda fuser went wrong, repro by running:\n"
                        f"\t{reproString(seed, args)}") from err


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-fuser", "--cuda_fuser", action='store_true', default=True)
    parser.add_argument("--legacy-fuser", "--legacy_fuser", action='store_true', default=False)
    parser.add_argument("--profiling-executor", "--profiling_executor", action='store_true', default=False)
    parser.add_argument("--fp16", action='store_true', default=False)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--debug-print", "--debug_print", action='store_true', default=False)
    parser.add_argument("--debug-tensor", "--debug_tensor", action='store_true', default=False)
    parser.add_argument("--max-num-tensor", "--max_num_tensor", default=MAX_TENSOR, type=int)
    parser.add_argument("--max-tensor-dim", "--max_tensor_dim", default=MAX_TENSOR_DIM, type=int)
    parser.add_argument("--max-tensor-size", "--max_tensor_size", default=MAX_TENSOR_SIZE, type=int)
    parser.add_argument("--depth-factor", "--depth-factor", default=GRAPH_FACTOR, type=int)
    parser.add_argument("--seed", default=45589, type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--iterations", default=4, type=int)
    group.add_argument("--repro-run", "--repro_run", action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Register CUDA fuser
    if args.cuda_fuser:
        torch._C._jit_set_nvfuser_enabled(True)

    # Turn off legacy fuser
    if not args.legacy_fuser:
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)

    # Turn off profiling executor
    if not args.profiling_executor:
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)

    # factor sorta control the depth of the model
    GRAPH_FACTOR = args.depth_factor
    # debug print
    DEBUG_PRINT = args.debug_print

    if args.repro_run:
        runTest(args.seed, args)
    else:
        np.random.seed(args.seed)
        failing_repros = []
        for seed in np.random.randint(0, args.seed, args.iterations):
            try:
                runTest(seed, args)
            except Exception as e:
                failing_repros.append(str(e))
        if len(failing_repros) == 0:
            print("test passed")
        else:
            print("{0} out of {1} tests failed;".format(
                  len(failing_repros), args.iterations))
            print("To repro failing tests, run\n")
            for repro in failing_repros:
                print(repro)
