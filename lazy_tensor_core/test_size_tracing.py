import torch
import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as metrics
import traceback

unique_lines = set()

def print_stacktrace():
    global unique_lines
    traceback.print_stack()
    stack = traceback.extract_stack()
    stack = traceback.format_list(stack)
    if (len(stack) > 1):
        second_frame = str(stack[-2]).replace("/home/villedepommes/miniconda3/envs/pytorch1/bin/ipython", "")
        unique_lines.add(second_frame)

lazy_tensor_core._LAZYC._set_custom_printer(print_stacktrace)


d = torch.rand(3, 3, device="cuda", requires_grad=True)
a = d.detach().clone().to(device="lazy").requires_grad_(True)
c = torch.rand(1, 1, device="lazy")
b = a.view(c.size(1), 9) 
b.sum().backward()
print(a.grad.to(device="cpu").sum())
print("done!")

print(f"unique lines {len(unique_lines)} {id(unique_lines)}")
for l in unique_lines:
    print(l)
print("done")

