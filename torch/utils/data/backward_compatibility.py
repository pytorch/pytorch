import warnings

def worker_init_fn(worker_id):
    warnings.warn("Usage of backward_compatibility.worker_init_fn is deprecated"
                  " as DataLoader automatically applies sharding in every worker", stacklevel=TO_BE_DETERMINED)
