"""
Utils shared by different modes of quantization (eager/graph)
"""

def get_combined_dict(default_dict, additional_dict):
    d = default_dict.copy()
    for k, v in additional_dict.items():
        d[k] = v
    return d
