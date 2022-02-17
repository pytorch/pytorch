import torch
import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as metrics



a = torch.rand(4, 1, device="lazy")

dim0 = lazy_tensor_core._LAZYC._dynamic_size(a, 0)
dim1 = lazy_tensor_core._LAZYC._dynamic_size(a, 1)
#b = a.view(1, 4)
b = lazy_tensor_core._LAZYC._dynamic_view(a, [dim1, dim0])
print(lazy_tensor_core._LAZYC._get_ltc_tensors_text([b]))
#c = torch.rand(1, 4, device="lazy")
#d = b + c
ltm.mark_step()
#print(d.to(device="cpu"))

print(b.to(device="cuda"))


#c = a.view(4, 1)
#print(lazy_tensor_core._LAZYC._get_ltc_tensors_text([c]))
#a.add_(b)
#print(lazy_tensor_core._LAZYC._get_ltc_tensors_text([c]))
#print(lazy_tensor_core._LAZYC._get_ltc_tensors_text([a]))
