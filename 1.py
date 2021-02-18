import torch
import torch.utils.benchmark as benchmark_utils
import os
print(os.getpid())

a = [torch.rand(2, 2, device="cuda").to(torch.uint8) for _ in range(2)]
b = [torch.rand(2, 2, device="cuda").to(torch.bool) for _ in range(2)]
scalar = 3+5j

print("\nwas")
print(a[0].dtype)
print(b[0].dtype)

print("\nres")
res = [t1.div(t2) for t1, t2 in zip(a, b)]
print(res)
print(res[0].dtype)

print("\nforeach")
res = torch._foreach_div(a, b)
print(res)
print(res[0].dtype)

'''
def foo():
    res = [t.add_(scalar) for t in a]

def foo2():
    res_fe = torch._foreach_add(a, scalar)

def main():
    timer = benchmark_utils.Timer(
        stmt="torch.cuda.synchronize(); foo()",
        globals=globals(),
        label="torch",
    )
    print(f"autorange:\n{timer.blocked_autorange()}\n\n")

    timer = benchmark_utils.Timer(
        stmt="torch.cuda.synchronize(); foo2()",
        globals=globals(),
        label="fe",
    )
    print(f"autorange:\n{timer.blocked_autorange()}\n\n")

if __name__ == "__main__":
    main()
'''

'''
import os 

print(os.getpid())
scalar = 2
dtype = torch.uint8

a = torch.rand(1, 5, device="cuda").to(dtype)
print(a.dtype)
res = a.mul(scalar)
print(res)
print(res.dtype)

print("\n\n\n")
fe_res = torch._foreach_mul([a], scalar)
print(fe_res)
print(fe_res[0].dtype)
'''