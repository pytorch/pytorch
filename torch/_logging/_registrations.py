from ._internal import register_artifact, register_log


TORCHDYNAMO_LOG_NAME = "torch._dynamo"
TORCHINDUCTOR_LOG_NAME = "torch._inductor"
AOT_AUTOGRAD_LOG_NAME = "torch._functorch.aot_autograd"

# (optional) shorthand names used in the logging env var and user API
TORCHDYNAMO_NAME = "dynamo"
TORCHINDUCTOR_NAME = "inductor"
AOT_AUTOGRAD_NAME = "aot"

# (optional) register log with shorthand name
register_log(TORCHDYNAMO_NAME, TORCHDYNAMO_LOG_NAME)
register_log(AOT_AUTOGRAD_NAME, AOT_AUTOGRAD_LOG_NAME)
register_log(TORCHINDUCTOR_NAME, TORCHINDUCTOR_LOG_NAME)

register_artifact("guards")
register_artifact("bytecode")
register_artifact("graph")
register_artifact("graph_code")
register_artifact("aot_forward_graph")
register_artifact("aot_backward_graph")
register_artifact("aot_joint_graph")
register_artifact("output_code", off_by_default=True)
register_artifact("schedule", off_by_default=True)
