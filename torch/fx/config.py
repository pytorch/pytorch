# Whether to disable showing progress on compilation passes
# Need to add a new config otherwise will get a circular import if dynamo config is imported here
disable_progress = True

# If True this also shows the node names in each pass, for small models this is great but larger models it's quite noisy
verbose_progress = False

# When True, skip collecting stack traces during tracing. This avoids the cost
# of CapturedTraceback.extract() and symbolization for every FX node, but means
# node.meta["stack_trace"] will be unset, degrading error messages and debugging info.
do_not_emit_stack_traces = False
