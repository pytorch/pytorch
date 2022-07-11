from .criterions import cel
from .ddp_models import basic_ddp_model
from .hook_states import BasicHookState
from .hooks import allreduce_hook, hybrid_hook, rpc_hook, sparse_rpc_hook
from .iteration_steps import basic_iteration_step
from .preprocess_data import preprocess_dummy_data
from .trainer import DdpTrainer

criterion_map = {
    "cel": cel
}

ddp_hook_map = {
    "allreduce_hook": allreduce_hook,
    "hybrid_hook": hybrid_hook,
    "rpc_hook": rpc_hook,
    "sparse_rpc_hook": sparse_rpc_hook
}

ddp_model_map = {
    "basic_ddp_model": basic_ddp_model
}

iteration_step_map = {
    "basic_iteration_step": basic_iteration_step
}

preprocess_data_map = {
    "preprocess_dummy_data": preprocess_dummy_data
}

hook_state_map = {
    "BasicHookState": BasicHookState
}

trainer_map = {
    "DdpTrainer": DdpTrainer
}
