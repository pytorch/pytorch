import csv
import torch
from collections import defaultdict


def get_xformer_results():
    """Returns a dictionary of benchmark results from the xformer repo""" 

    data = defaultdict(list)

    # Mapping from string to actual PyTorch dtype
    dtype_map = {
        'torch.float16': torch.float16,
        'torch.bfloat16': torch.bfloat16,
        'torch.float32': torch.float32,
        'torch.float': torch.float,
        'torch.double': torch.double
    }

    with open('/home/joydong/.cache/xformers/benchmarks/attn_decoding/optimized.NVIDIA_H100.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:

            sub_label, label, num_threads, algorithm, description, runtime_us, mem_use_mb = [row[key] for key in ['sub_label', 'label', 'num_threads', 'algorithm', 'description', 'runtime_us', 'mem_use_mb']]
            params = sub_label.split(' ')
            
            # Extracting values from the formatted strings
            B = int(params[0].split('=')[1])
            Mq = int(params[1].split('=')[1])
            Mkv = int(params[2].split('=')[1])
            Hq = int(params[3].split('=')[1])
            Hkv = int(params[4].split('=')[1])
            K = int(params[5].split('=')[1])
            dtype_str = params[6].split('=')[1]
            dtype = dtype_map[dtype_str]  # Convert string to actual dtype

            if algorithm != 'triton_splitK':
                continue          
            data[(B, Mkv, Hq, Hkv, K, dtype)] = runtime_us

    return data
