import torch

exported = torch.export.load("mm_model.pt2")

torch._inductor.aoti_compile_and_package(
    exported, 
    package_path="aoti_mm_model.pt2", 
)
print("exported")
