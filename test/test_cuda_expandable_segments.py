# run time cuda tests, but with the allocator using expandable segments

import os
import torch

torch.cuda.memory._set_allocator_settings('expandable_segments:False')
current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, 'test_cuda.py')
exec(compile(open(filepath, 'r').read(), filepath, mode='exec'))
