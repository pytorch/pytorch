
from microbenchmarks.operator_inp_utils import OperatorInputsLoader
from torch.utils._ordered_set import OrderedSet
loader = OperatorInputsLoader.get_huggingface_loader()
import os
import torch
aten = torch.ops.aten

import json
def load_cached_shapes(op, dtype):
    """Load previously processed shapes from cache file."""
    cache_file = f"shapes_cache_{op}_{dtype}.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                return set(tuple(shape) for shape in json.load(f))
        except (json.JSONDecodeError, FileNotFoundError):
            return set()
    raise Exception(f"Something went wrong with {cache_file}")


def save_cached_shapes(op, dtype, shapes_set):
    """Save processed shapes to cache file."""
    cache_file = f"shapes_cache_{op}_{dtype}.json"
    with open(cache_file, "w") as f:
        json.dump(list(shapes_set), f)


def extract_shapes_from_loader(op, dtype, processed_shapes):
    """Extract M,K,N shapes from loader, skipping already processed ones."""
    new_shapes = []

    print(f"Extracting shapes from loader for {op} {dtype}...")
    for i, (args, kwargs) in enumerate(
        loader.get_inputs_for_operator(op, dtype=dtype, device="cuda")
    ):
        try:
            inp_t = args[1]
            weight_t = args[2]
        except:
            inp_t = args[0]
            weight_t = args[1]

        if len(inp_t.shape) != 2:
            continue

        if inp_t.numel() == 0:
            continue

        M, K, N = inp_t.shape[0], inp_t.shape[1], weight_t.shape[1]
        shape_tuple = (M, K, N)

        if shape_tuple not in processed_shapes:
            new_shapes.append((i, args, kwargs, shape_tuple))
            processed_shapes.add(shape_tuple)
            print(f"Found new shape: {M}_{K}_{N}")

    return new_shapes

shape_set = set([])
new_shapes = extract_shapes_from_loader(aten.mm.default, torch.float16, shape_set)
print(new_shapes)
save_cached_shapes(aten.mm.default, torch.float16, shape_set)
loaded = load_cached_shapes(aten.mm.default, torch.float16)
print(loaded)
