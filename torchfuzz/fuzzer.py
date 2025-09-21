
import time
import secrets
import random
import argparse
import os
from codegen import CodeGenerator
from runner import ProgramRunner

class Fuzzer:
    def __init__(self, supported_ops, max_depth, seed):
        self.supported_ops = supported_ops  # List of Operator instances
        self.max_depth = max_depth
        random.seed(seed)
        self.generated_nodes = []  # List of all nodes (for dependency tracking)

    def fuzz(self, output_path=None):
        # Generate a random nonce for the filename
        nonce = secrets.token_hex(8)
        if output_path is None:
            output_path = f"/tmp/torchfuzz/fuzz_{nonce}.py"
        else:
            # If output_path is provided, override it to match the new requirement
            output_path = f"/tmp/torchfuzz/fuzz_{nonce}.py"

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        target = self.generate_random_tensor()
        # Each node is a tuple: (output_tensor, op, input_tensors)
        # For the initial tensor, op and input_tensors are None
        nodes = [(target, None, [])]
        all_tensors = {id(target): target}
        tensor_to_node = {id(target): nodes[0]}

        # Maintain a pool of previously generated tensors to allow sharing
        tensor_pool = [target]

        for _ in range(self.max_depth):
            new_nodes = []
            for node in nodes:
                output_tensor, op, inputs = node
                # Only decompose leaf tensors (those not produced by an op)
                if op is None and self.supported_ops:
                    # Decompose tensor into op and its input tensors
                    candidates = [op for op in self.supported_ops if op.can_produce(output_tensor)]
                    if not candidates:
                        continue
                    chosen_op = random.choice(candidates)
                    # Get the "ideal" input tensors for this op
                    # --- CHANGED: pass random number of inputs to decompose ---
                    if isinstance(chosen_op, AddOperator) or isinstance(chosen_op, CatOperator):
                        min_inputs = 2
                        max_inputs = 5
                        num_inputs = random.randint(min_inputs, max_inputs)
                        ideal_inputs = chosen_op.decompose(output_tensor, num_inputs=num_inputs)
                    else:
                        ideal_inputs = chosen_op.decompose(output_tensor)
                    input_tensors = []
                    for t in ideal_inputs:
                        # With some probability, try to share a compatible tensor from the pool
                        if len(tensor_pool) > 1 and random.random() < 0.5:
                            # Exclude the current output_tensor to avoid trivial cycles
                            # Also exclude any tensors that would create circular dependencies
                            candidates_for_sharing = []
                            for tp in tensor_pool:
                                if (tp is not output_tensor
                                    and tp.size == t.size
                                    and tp.stride == t.stride
                                    and tp.dtype == t.dtype
                                    and tp.device == t.device):
                                    # Check if using this tensor would create a circular dependency
                                    if not self._would_create_cycle(output_tensor, tp, tensor_to_node):
                                        candidates_for_sharing.append(tp)

                            if candidates_for_sharing:
                                shared_tensor = random.choice(candidates_for_sharing)
                                input_tensors.append(shared_tensor)
                                continue
                        input_tensors.append(t)
                        tensor_pool.append(t)
                        all_tensors[id(t)] = t
                    new_node = (output_tensor, chosen_op, input_tensors)
                    tensor_to_node[id(output_tensor)] = new_node
                    # Instead of just adding new leaves, always update tensor_to_node for all input_tensors
                    for t in input_tensors:
                        # Only add as a new leaf if not already present
                        if id(t) not in tensor_to_node:
                            tensor_to_node[id(t)] = (t, None, [])
                            new_nodes.append((t, None, []))
                    self.generated_nodes.append(new_node)
                else:
                    # Already decomposed or no supported ops
                    continue
            if not new_nodes:
                break
            # Enforce deterministic order for new_nodes to ensure topological order
            # Sort by id of the output tensor to have a consistent order
            new_nodes.sort(key=lambda n: id(n[0]))
            nodes = new_nodes

        # Collect all nodes (including leaves) and perform proper topological sort
        all_nodes = self._topological_sort(target, tensor_to_node)

        # Generate and write code using CodeGenerator
        codegen = CodeGenerator()
        codegen.generate_code(target, all_nodes, output_path)
        abs_path = os.path.abspath(output_path)
        print(f"Generated program written to: {abs_path}")

        # Run the generated program using ProgramRunner
        runner = ProgramRunner()
        runner.run_program(abs_path)

    def _topological_sort(self, target, tensor_to_node):
        """
        Perform proper topological sort with circular dependency detection.
        Returns list of nodes in dependency order (dependencies before uses).
        """
        all_nodes = []
        visited = set()
        visiting = set()  # For cycle detection

        def visit(tensor):
            tid = id(tensor)
            if tid in visited:
                return
            if tid in visiting:
                # Circular dependency detected - skip to avoid infinite loop
                print(f"Warning: Circular dependency detected involving tensor {tid}, skipping...")
                return

            visiting.add(tid)
            node = tensor_to_node.get(tid, (tensor, None, []))

            # Visit all input tensors first (dependencies)
            for input_tensor in node[2]:
                visit(input_tensor)

            # Then add this node
            all_nodes.append(node)
            visiting.remove(tid)
            visited.add(tid)

        visit(target)
        return all_nodes

    def _would_create_cycle(self, output_tensor, candidate_input, tensor_to_node):
        """
        Check if using candidate_input as an input to output_tensor would create a cycle.
        Returns True if it would create a cycle, False otherwise.
        """
        # Check if candidate_input depends on output_tensor
        def depends_on(tensor, target):
            if tensor is target:
                return True
            tid = id(tensor)
            node = tensor_to_node.get(tid)
            if node is None or node[1] is None:  # Leaf node
                return False
            # Check all inputs recursively
            for input_tensor in node[2]:
                if depends_on(input_tensor, target):
                    return True
            return False

        return depends_on(candidate_input, output_tensor)

    def generate_random_tensor(self):
        # Allow ndim=0 to generate a scalar tensor
        ndim = random.randint(0, 3)
        if ndim == 0:
            size = ()
        else:
            size = tuple(random.randint(1, 1024) for _ in range(ndim))

        stride = []
        acc = 1
        for s in reversed(size):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        dtypes = ['float32', 'float16', 'bfloat16']
        devices = ['cuda']
        dtype = random.choice(dtypes)
        device = random.choice(devices)

        return Tensor(size, stride, dtype, device, self.supported_ops)

class Tensor:
    def __init__(self, size, stride, dtype, device, supported_ops):
        self.size = size
        self.stride = stride
        self.dtype = dtype
        self.device = device
        self.supported_ops = supported_ops
        # Add optional attributes for cat operation - allow int values
        self._cat_dim = None  # type: int | None
        self._cat_sizes = None  # type: tuple | None
        # For view/sum
        self._view_shape = None  # type: tuple | None
        self._sum_dim = None  # type: int | tuple | None

    def decompose(self):
        candidates = []
        for op in self.supported_ops:
            if op.can_produce(self):
                candidates.append(op)
        if not candidates:
            return []
        candidate = random.choice(candidates)
        # --- CHANGED: pass random number of inputs for Add/Cat ---
        if isinstance(candidate, AddOperator) or isinstance(candidate, CatOperator):
            min_inputs = 2
            max_inputs = 5
            num_inputs = random.randint(min_inputs, max_inputs)
            return candidate.decompose(self, num_inputs=num_inputs)
        return candidate.decompose(self)

class Operator:
    def __init__(self, name):
        self.name = name

    def can_produce(self, tensor):
        raise NotImplementedError

    def decompose(self, tensor):
        raise NotImplementedError

    def codegen(self, output_name, input_names, output_tensor):
        raise NotImplementedError

class AddOperator(Operator):
    def __init__(self):
        super().__init__("add")

    def can_produce(self, tensor):
        # Add can always produce a tensor by adding two tensors of the same shape, dtype, etc.
        return True

    def decompose(self, tensor, num_inputs=2):
        # Type promotion table for realistic LLM/diffusion model types
        # Each output dtype maps to possible input dtype pairs (in order of preference)
        promotion_table = {
            "float32": [
                ("float32", "float32"),
                ("bfloat16", "float32"),
                ("float32", "bfloat16"),
                ("bfloat16", "bfloat16"),
                ("float16", "float32"),
                ("float32", "float16"),
                ("float16", "float16"),
            ],
            "bfloat16": [
                ("bfloat16", "bfloat16"),
                ("float16", "bfloat16"),
                ("bfloat16", "float16"),
            ],
            "float16": [
                ("float16", "float16"),
            ],
        }
        # If num_inputs > 2, promote left-to-right (e.g. (((a + b) + c) + d))
        # For simplicity, we generate the first two with promotion, rest match output dtype
        dtype = tensor.dtype
        supported_types = promotion_table.get(dtype, [(dtype, dtype)])
        # Pick a random promotion pattern for the first two inputs
        if num_inputs >= 2:
            dtypes = list(random.choice(supported_types))
            # For >2 inputs, fill with output dtype
            while len(dtypes) < num_inputs:
                dtypes.append(dtype)
        else:
            dtypes = [dtype] * num_inputs

        return [
            Tensor(tensor.size, tensor.stride, dt, tensor.device, tensor.supported_ops)
            for dt in dtypes
        ]


    def codegen(self, output_name, input_names, output_tensor):
        # Sum all input tensors
        expr = " + ".join(input_names)
        return f"{output_name} = {expr}"

class CatOperator(Operator):
    def __init__(self):
        super().__init__("cat")

    def can_produce(self, tensor):
        # Can only cat if there is at least one dimension with size >= 2
        return any(s >= 2 for s in tensor.size)

    def decompose(self, tensor, num_inputs=2):
        # Find all candidate dimensions where size is at least 2
        candidate_dims = [i for i, s in enumerate(tensor.size) if s >= 2]
        if not candidate_dims:
            # No suitable dimension to split, fallback to single tensor
            return [
                Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
                for _ in range(num_inputs)
            ]

        # Randomly select one of the candidate dimensions
        dim = random.choice(candidate_dims)

        # Randomly choose split points to divide the dimension into num_inputs parts
        total = tensor.size[dim]
        # Ensure each part has at least size 1
        if num_inputs > total:
            num_inputs = total
        # Generate split points
        splits = sorted(random.sample(range(1, total), num_inputs - 1))
        sizes = []
        prev = 0
        for s in splits + [total]:
            sizes.append(s - prev)
            prev = s

        # Build size tuples for each input tensor
        input_sizes = [
            tensor.size[:dim] + (sz,) + tensor.size[dim+1:]
            for sz in sizes
        ]

        # Calculate proper strides for the split tensors
        def calculate_stride(size):
            stride = []
            acc = 1
            for s in reversed(size):
                stride.insert(0, acc)
                acc *= s
            return tuple(stride)

        input_strides = [calculate_stride(sz) for sz in input_sizes]

        tensors = [
            Tensor(sz, st, tensor.dtype, tensor.device, tensor.supported_ops)
            for sz, st in zip(input_sizes, input_strides)
        ]

        # Store the cat dimension on the output tensor instead of input tensors
        tensor._cat_dim = dim
        tensor._cat_sizes = tuple(input_sizes)

        return tensors

    def codegen(self, output_name, input_names, output_tensor):
        # Try to find the dimension along which to cat
        # If the tensor has attribute _cat_dim, use it; else, pick the first valid
        dim = getattr(output_tensor, "_cat_dim", None)
        if dim is None:
            dim = next((i for i, s in enumerate(output_tensor.size) if s >= 2), 0)
        return f"{output_name} = torch.cat([{', '.join(input_names)}], dim={dim})"

class ViewOperator(Operator):
    def __init__(self):
        super().__init__("view")

    def can_produce(self, tensor):
        # View can always target any shape with the same numel.
        return True

    def decompose(self, tensor):
        # Pick an input shape with the same numel as tensor.size
        numel = 1
        for s in tensor.size:
            numel *= s

        # Try to factor numel into 1-3 dims
        for _ in range(10):
            ndims = random.randint(1, 3)

            def random_shape(n, d):
                if d == 1:
                    return (n,)
                factors = []
                rem = n
                for i in range(d - 1):
                    divisors = [f for f in range(1, rem + 1) if rem % f == 0]
                    f = random.choice(divisors)
                    factors.append(f)
                    rem //= f
                factors.append(rem)
                return tuple(factors)

            shape = random_shape(numel, ndims)
            if all(isinstance(x, int) and x > 0 for x in shape):
                break
        else:
            shape = (numel,)

        # contiguous stride for input
        stride = []
        acc = 1
        for s in reversed(shape):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        t_in = Tensor(shape, stride, tensor.dtype, tensor.device, tensor.supported_ops)
        # No metadata needed on inputs. Codegen will view to the output size.
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        # Always view to the output's shape
        return f"{output_name} = {input_names[0]}.view({tuple(output_tensor.size)})"

class SumOperator(Operator):
    def __init__(self):
        super().__init__("sum")

    def can_produce(self, tensor):
        # We construct inputs by inserting at most one extra dimension,
        # so we need room to add a dim and stay within a reasonable cap.
        # Your generator uses up to 5 dims, so keep input_dim <= 5.
        return len(tensor.size) < 5

    def decompose(self, tensor):
        """
        Construct an input shape that reduces to tensor.size via a sum.
        Store the chosen reduction dims on the OUTPUT tensor so codegen can read it.
        """
        if len(tensor.size) == 0:
            # Scalar output, pick an arbitrary input and reduce all dims.
            input_ndim = random.randint(1, 3)
            input_shape = tuple(random.randint(2, 5) for _ in range(input_ndim))
            # Mark 'all' to emit .sum() with no dim argument.
            tensor._sum_dim = "all"
        else:
            # Insert a new dimension of size >= 2 at a random position,
            # then reduce over that single dimension.
            dim = random.randint(0, len(tensor.size))
            expand_size = random.randint(2, 5)
            input_shape = list(tensor.size)
            input_shape.insert(dim, expand_size)
            input_shape = tuple(input_shape)
            tensor._sum_dim = dim

        # contiguous stride for input
        stride = []
        acc = 1
        for s in reversed(input_shape):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        t_in = Tensor(input_shape, stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        sd = getattr(output_tensor, "_sum_dim", None)
        src = input_names[0]
        if sd == "all":
            return f"{output_name} = {src}.sum()"
        elif isinstance(sd, tuple):
            # If you later extend to multi-dim reductions, this handles it.
            return f"{output_name} = {src}.sum(dim={sd})"
        elif isinstance(sd, int):
            return f"{output_name} = {src}.sum(dim={sd})"
        else:
            # Safe default for legacy cases: reduce all dims
            return f"{output_name} = {src}.sum()"

class GeluOperator(Operator):
    def __init__(self):
        super().__init__("gelu")

    def can_produce(self, tensor):
        # GELU can be applied to any tensor (elementwise op)
        return True

    def decompose(self, tensor):
        # The input to GELU must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        # Use torch.nn.functional.gelu for the GELU activation
        return f"{output_name} = torch.nn.functional.gelu({input_names[0]})"

class FillDiagonalOperator_(Operator):
    def __init__(self):
        super().__init__("fill_diagonal_")

    def can_produce(self, tensor):
        # PyTorch's fill_diagonal_ requires all dimensions to be of equal length
        # Only produce for tensors where all dimensions have the same size
        if len(tensor.size) < 2:
            return False
        # Check if all dimensions have the same size
        first_size = tensor.size[0]
        return all(s == first_size for s in tensor.size)

    def decompose(self, tensor):
        # Import here to avoid circular import
        from tensor import Tensor

        # Find two dimensions with equal size
        dims = None
        for i in range(len(tensor.size)):
            for j in range(i + 1, len(tensor.size)):
                if tensor.size[i] == tensor.size[j]:
                    dims = (i, j)
                    break
            if dims is not None:
                break
        if dims is None:
            # Fallback: just return a tensor of the same shape and a scalar
            return [
                Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops),
                Tensor((), (), tensor.dtype, tensor.device, tensor.supported_ops)
            ]

        # Input 0: tensor to fill (same shape as output)
        # Input 1: value to fill (scalar)
        t_in = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        t_val = Tensor((), (), tensor.dtype, tensor.device, tensor.supported_ops)
        # Store dims for codegen
        tensor._fill_diag_dims = dims
        return [t_in, t_val]

    def codegen(self, output_name, input_names, output_tensor):
        # PyTorch's fill_diagonal_ expects a scalar number, so use .item() to extract from tensor
        fill_value = f"{input_names[1]}.item()"
        # Since all dimensions are equal now, we can just use fill_diagonal_ directly
        return f"{output_name} = {input_names[0]}.clone(); {output_name}.fill_diagonal_({fill_value})"

def main():
    parser = argparse.ArgumentParser(description="Fuzzer for generating PyTorch programs.")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of the operation tree.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If not set, a random seed will be generated.")
    parser.add_argument("--output", type=str, default=None, help="Output file path.")
    args = parser.parse_args()

    supported_ops = [AddOperator(), CatOperator(), ViewOperator(), SumOperator(), GeluOperator(), FillDiagonalOperator_()]
    max_depth = args.max_depth
    if args.seed is not None:
        seed = args.seed
    else:
        # Use a combination of time and secrets for randomness
        seed = int(time.time() * 1000) ^ secrets.randbits(32)
        print(f"No seed provided, generated random seed: {seed}")
    output_path = args.output
    print(f"Running fuzzer with max_depth={max_depth}, seed={seed}, output={output_path}")
    fuzzer = Fuzzer(supported_ops, max_depth, seed)
    fuzzer.fuzz(output_path=output_path)


if __name__ == "__main__":
    main()
