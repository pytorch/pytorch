import torch


GreenContext = None

if hasattr(torch._C, "GreenContext"):
    GreenContext = torch._C.GreenContext
