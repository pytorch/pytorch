
import time
import secrets
import random
import argparse
import os
from codegen import CodeGenerator
from runner import ProgramRunner
from operators import *  # Import everything from operators
from tensor import Tensor

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
                    if chosen_op.supports_variable_inputs():
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
                                    and tp.device == t.device
                                    and getattr(tp, 'requires_grad', None) == getattr(t, 'requires_grad', None)
                                    # Don't share batch_norm buffer tensors
                                    and not getattr(t, '_batch_norm_buffer', False)
                                    and not getattr(tp, '_batch_norm_buffer', False)):
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
            size = tuple(random.randint(1, 128) for _ in range(ndim))

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

def main():
    parser = argparse.ArgumentParser(description="Fuzzer for generating PyTorch programs.")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of the operation tree.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If not set, a random seed will be generated.")
    parser.add_argument("--output", type=str, default=None, help="Output file path.")
    args = parser.parse_args()

    # Dynamically initialize all operator classes imported from operators
    import operators
    supported_ops = []
    for name in dir(operators):
        obj = getattr(operators, name)
        if isinstance(obj, type) and name != "Operator":
            supported_ops.append(obj())
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
