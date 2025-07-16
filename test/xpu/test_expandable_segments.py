import os
import torch

# Owner(s): ["module: xpu"]
# run time xpu tests, but with the allocator using expandable segments

if torch.xpu.is_available():
    torch.xpu.memory._set_allocator_settings('expandable_segments:True')

current_dir = os.path.dirname(os.path.abspath(__file__))

current_file = os.path.basename(__file__)

for fname in os.listdir(current_dir):
    if fname.endswith('.py') and fname != current_file:
        file_path = os.path.join(current_dir, fname)
        print(f"Running {file_path} ...")
        exec(compile(open(file_path, 'r').read(), file_path, mode='exec'))
