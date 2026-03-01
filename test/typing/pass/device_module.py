import torch


torch.register_module_for_device("cpu", torch.cpu)
torch.register_module_for_device("cuda", torch.cuda)
torch.register_module_for_device("mps", torch.mps)
torch.register_module_for_device("mtia", torch.mtia)
torch.register_module_for_device("xpu", torch.xpu)
