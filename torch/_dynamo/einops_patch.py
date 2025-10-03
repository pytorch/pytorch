import einops

try:
    einops.einops._reconstruct_from_shape = (
        einops.einops._reconstruct_from_shape_uncached
    )
    einops.einops._prepare_transformation_recipe = (
        einops.einops._prepare_transformation_recipe.__wrapped__
    )
except Exception:
    pass
