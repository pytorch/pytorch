import torch
from torch._C._dynamo import guards


def print_type_guard_failure(x):
    return f"expected ? but found {type(x)}"


def print_const_guard_failure(x):
    return f"expected ? but found {x}"


# Check LeafGuard formation
const_guard = guards.PythonLambdaGuard(lambda x: x == 5, print_type_guard_failure)
print(const_guard(4))
print(const_guard(5))

# Check GuardManager and a leaf guard
guard_manager = guards.GuardManager()
guard_manager.add_lambda_guard(lambda x: isinstance(x, int), print_type_guard_failure)
guard_manager.add_lambda_guard(lambda x: x >= 5, print_const_guard_failure)

print([guard_manager.check(i) for i in range(4, 8)])


# Check GuardManager with an attr and leaf guards on attr
guard_manager.animesh.add_lambda_guard(
    lambda x: isinstance(x, int), print_type_guard_failure
)
guard_manager.animesh.add_lambda_guard(lambda x: x <= 10, print_const_guard_failure)
guard_manager.animesh.add_lambda_guard(lambda x: x >= 5, print_const_guard_failure)
print(guard_manager.animesh.check(9), guard_manager.animesh.check(11))


# Check GuardManager with an item and leaf guards on item
guard_manager["foo"].add_lambda_guard(
    lambda x: isinstance(x, int), print_type_guard_failure
)
guard_manager["foo"].add_lambda_guard(lambda x: x <= 10, print_const_guard_failure)
guard_manager["foo"].add_lambda_guard(lambda x: x >= 5, print_const_guard_failure)
print(guard_manager["foo"].check(9), guard_manager["foo"].check(11))


# Put everything together - f_locals is a dictionary (__getitem__). f_locals["bar"] is a pair of x, y


class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class PairImpostor:
    def __init__(self, x, y):
        self.x = x
        self.y = y

f_locals = {
    "foo": 5,
    "bar": Pair(1, 2),
}
guard_manager = guards.GuardManager()

guard_manager["foo"].add_lambda_guard(
    lambda x: isinstance(x, int), print_type_guard_failure
)
guard_manager["foo"].add_lambda_guard(lambda x: x == 5, print_const_guard_failure)
# Just add same guard to test if guard reshuffling happens on failure
guard_manager["foo"].add_lambda_guard(lambda x: x == 5, print_const_guard_failure)
guard_manager["foo"].add_lambda_guard(lambda x: x == 5, print_const_guard_failure)

guard_manager["bar"].add_lambda_guard(
    lambda x: isinstance(x, Pair), print_type_guard_failure
)
guard_manager["bar"].x.add_lambda_guard(
    lambda x: isinstance(x, int), print_type_guard_failure
)
guard_manager["bar"].x.add_lambda_guard(lambda x: x == 1, print_const_guard_failure)
guard_manager["bar"].y.add_lambda_guard(
    lambda x: isinstance(x, int), print_type_guard_failure
)
guard_manager["bar"].y.add_lambda_guard(lambda x: x == 2, print_const_guard_failure)

f_locals1 = {
    "foo": 5,
    "bar": Pair(1, 3),
}

print(
    guard_manager.check_with_debug_info(f_locals),
    # fails for the first time, reshuffles the guard order and brings the failing guard to the top
    guard_manager.check_with_debug_info(f_locals1),
    guard_manager.check_with_debug_info(f_locals1),  # This time it fails much faster
)


def fn(x, foo, bar):
    return x * foo * bar.x * bar.y

opt_fn = torch.compile(fn, backend="eager")
opt_fn(torch.randn(4), 5, Pair(1, 2))
opt_fn(torch.randn(4), 5, Pair(1, 2))
opt_fn(torch.randn(4), 5, Pair(1, 2))
print("---", flush=True)
opt_fn(torch.randn(4), 5, PairImpostor(1, 2))
