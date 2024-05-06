import torch
import torch.utils.benchmark as benchmark

MEMO = {}


def create_nested_dict_type(layers):
    if layers == 0:
        return torch._C.StringType.get()
    if layers not in MEMO:
        less_nested = create_nested_dict_type(layers - 1)
        result = torch._C.DictType(
            torch._C.StringType.get(), torch._C.TupleType([less_nested, less_nested])
        )
        MEMO[layers] = result
    return MEMO[layers]


nesting_levels = (1, 3, 5, 10)
types = (reasonable, medium, big, huge) = [
    create_nested_dict_type(x) for x in nesting_levels
]

timers = [
    benchmark.Timer(stmt="x.annotation_str", globals={"x": nested_type})
    for nested_type in types
]

for nesting_level, typ, timer in zip(nesting_levels, types, timers):
    print("Nesting level:", nesting_level)
    print("output:", typ.annotation_str[:70])
    print(timer.blocked_autorange())
