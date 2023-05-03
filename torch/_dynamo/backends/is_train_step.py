class TrainStepCompiler:
    def __init__(self, compile_fn):
        self.compile_fn = compile_fn

    def __call__(self, *args, **kwargs):
        return self.compile_fn(*args, **kwargs)


def _is_train_step_compiler(compiler_fn):
    return isinstance(compiler_fn, TrainStepCompiler)
