import torch._inductor.config as config

def gen_best_config(mat1, mat2):
    """
    Generate the best GEMM autotune config for the given matrices.
    """
    if config.is_fbcode():
        from torch._inductor.fb.remote_gemm_autotune_cache import gen_best_config
        gen_best_config(mat1, mat2)
    else:
        raise NotImplementedError("Function gen_best_config is not yet implemented")
