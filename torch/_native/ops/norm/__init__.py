from .flydsl_rmsnorm_bwd_impl import (
    register_op_override as register_flydsl_rmsnorm_bwd_overrides,
)
from .rmsnorm_impl import register_rmsnorm_overrides


register_rmsnorm_overrides()
register_flydsl_rmsnorm_bwd_overrides()
