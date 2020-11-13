
'''
import torch
import time
d1 = torch.int32
d2 = torch.int64
r = 1
scalars = [1.1 for _ in range(r)]
tensors = [torch.rand(2, 2, dtype=torch.float64, device="cuda") for _ in range(r)]
exp = [t.add(s) for t, s in zip(tensors, scalars)]
print(exp)
print(exp[0].dtype)

res = torch._foreach_add_sl(tensors, scalars)
print(res)
print(res[0].dtype)
'''


import torch
import torch.utils.benchmark as benchmark_utils

r = 1
scalars = [1.1 for _ in range(r)]
tensors = [torch.rand(2, 2, dtype=torch.float64, device="cuda") for _ in range(r)]

exp = [t.div(s) for t, s in zip(tensors, scalars)]
res = torch._foreach_div_sl(tensors, scalars)
torch._foreach_div_sl_(tensors, scalars)

print(res)
print(tensors)
'''
def main():
    timer = benchmark_utils.Timer(
        stmt="exp = [t.add(s) for t, s in zip(tensors, scalars)]",
        globals=globals(),
        label="str(optimizer)",
    )
    print(f"autorange:\n{timer.blocked_autorange()}\n\n")

    timer_mta = benchmark_utils.Timer(
        stmt="res = torch._foreach_add_sl(tensors, scalars)",
        globals=globals(),
        label="str(optimizer_mta)",
    )
    print(f"autorange:\n{timer_mta.blocked_autorange()}\n\n")

if __name__ == "__main__":
    main()
'''