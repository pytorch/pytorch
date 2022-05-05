import torch._C._lazy

def dump(dot_file_name: str):
    """Return a text dump of the TrieCache in dot format"""
    return torch._C._lazy._dump_ir_cache(dot_file_name)

def reset():
    """Clear the TrieCache. This is needed in testing to avoid
    node reusing between test cases.
    """
    return torch._C._lazy._clear_ir_cache()
