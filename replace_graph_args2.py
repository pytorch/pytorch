import torch

x = torch.zeros(8, device="cuda")
y = x * 2

g1 = torch.cuda.CUDAGraph()
g1.enable_debug_mode()
g2 = torch.cuda.CUDAGraph()
g2.enable_debug_mode()

x_static1 = torch.zeros(8, device="cuda")

with torch.cuda.graph(g1):
    y_static1 = x_static1 * 2

x_static2 = torch.zeros(8, device="cuda")

with torch.cuda.graph(g2):
    y_static2 = x_static2 * 2

updates = torch.create_device_updates(g1, g2, [(x_static1, x_static2), (y_static1, y_static2)])

print(updates)

g1.debug_dump("g1.dot")
g2.debug_dump("g2.dot")

# import ipdb; ipdb.set_trace()
