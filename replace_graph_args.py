import torch

x = torch.zeros(8, device="cuda")
y = torch.ones(8, device="cuda")

z = x + y

g1 = torch.cuda.CUDAGraph()
g1.enable_debug_mode()
g2 = torch.cuda.CUDAGraph()
g2.enable_debug_mode()

x_static1 = torch.zeros(8, device="cuda")
y_static1 = torch.ones(8, device="cuda")

with torch.cuda.graph(g1):
    z_static1 = x_static1 + y_static1

x_static2 = torch.zeros(8, device="cuda")
y_static2 = torch.ones(8, device="cuda")

with torch.cuda.graph(g2):
    z_static2 = x_static2 + y_static2

# NOTE: This has a false positive for me right now. x_static1.data_ptr - x_static2.data_ptr == y_static1.data_ptr - y_static2.data_ptr
updates = torch.create_device_updates(g1, g2, [(x_static1, x_static2), (y_static1, y_static2), (z_static1, z_static2)])

print(updates)

g1.debug_dump("g1.dot")
g2.debug_dump("g2.dot")

# import ipdb; ipdb.set_trace()
