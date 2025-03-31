import torch
from torch._inductor.compile_fx import compile_fx
import contextlib
from torch._inductor.runtime.cache_dir_utils import cache_dir
from unittest.mock import patch

# =============== APIs ===================

class CompiledArtifact:
    # private ctor
    def __init__(self, compiled_fn, artifact_bytes, was_loaded):
        self.compiled_fn = compiled_fn
        self.artifact_bytes = artifact_bytes
        self.was_loaded = was_loaded

    def __call__(self, *args):
        return self.compiled_fn(*args)

    def save(self, filename):
        with open(filename, 'wb') as file:
            file.write(self.artifact_bytes)

    @staticmethod
    def load(filename, *example_args):
        with open(filename, 'rb') as file:
            artifact_bytes = file.read()
        with fresh_compile_context():
            torch.compiler.load_cache_artifacts(artifact_bytes)
            # TODO: There should be an easier way to
            # "get the only item from the AOTAutograd cache"

            # Get the only item in the AOTAutograd Cache
            from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
            import os

            folder = AOTAutogradCache._get_tmp_dir()
            items = os.listdir(folder)
            keys = [item for item in items if os.path.isdir(os.path.join(folder, item))]
            assert len(keys) == 1
            key = keys[0]

            with torch._functorch.config.patch(strict_autograd_cache=True):
                entry = AOTAutogradCache._lookup(key, local=True, remote=False)

            assert entry is not None

            # TODO(rzou): We shouldn't need the configs?
            # What are they used for?
            # Can these be a part of the serialized thing?
            aot_config = torch._dynamo.variables.higher_order_ops.get_dummy_aot_autograd_config()
            fx_config = {
                "cudagraphs": False,
                "boxed_forward_device_index": 0,
            }

            # TODO: shouldn't need example args.
            # Instead, inductor_compile should error if it specializes
            # on SymInts.
            with contextlib.ExitStack() as exit_stack:
                exit_stack.enter_context(
                    patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                          lambda *args, **kwargs: AlwaysHitShapeEnv()))

                # TODO: had to comment out a lot of the logging code.
                compiled_fn = entry.wrap_post_compile(example_args, aot_config, fx_config)

                # Why the calling convention change?
                def actual_compiled_fn(*args):
                    return compiled_fn(list(args))

            # TODO: Keep around the Info struct, makes it easier to debug.
            return CompiledArtifact(actual_compiled_fn, None, was_loaded=True)

def inductor_compile(gm, example_inputs, **kwargs):
    # TODO: Does it make sense to provide this API?
    # The problem is Dynamic Shapes and the SymInt input.
    symint = example_inputs[2]
    shape_env = symint.node.shape_env
    from torch._subclasses import FakeTensorMode
    torch._guards._TLS.tracing_context = torch._guards.TracingContext(FakeTensorMode(shape_env=shape_env))

    with fresh_compile_context():
        compiled_fn = compile_fx(gm, example_inputs, **kwargs)
        artifact_bytes, info = torch.compiler.save_cache_artifacts()

        # inference-only: one inductor artifact. But can be 2 for training
        assert len(info.inductor_artifacts) == 1
        assert len(info.aot_autograd_artifacts) == 1

        return CompiledArtifact(compiled_fn, artifact_bytes, was_loaded=False)

from torch._inductor.utils import fresh_inductor_cache

@contextlib.contextmanager
def fresh_compile_context():
    with fresh_inductor_cache(), torch._dynamo.utils.get_metrics_context():
        yield

def capture(fn):
    def inner(*args):
        gm = None
        actual_args = None
        kwargs = None

        def backend(gm_, args_, **kwargs_):
            nonlocal gm
            nonlocal actual_args
            nonlocal kwargs
            gm = gm_
            actual_args = args_
            kwargs = kwargs_
            return gm

        _ = torch.compile(fn, fullgraph=True, backend=backend)(*args)
        return gm, actual_args, kwargs
    return inner

class AlwaysHitShapeEnv:
    """
    Why do we need this class:

    For normal `torch.compile` usage, every compilation will have
    one Dynamo bytecode compilation and one Inductor compilation.
    The Inductor compilation happens under the context of the
    Dynamo bytecode compilation, and that context is used to
    determine the dynamic shape information, etc.

    For our use case, we only run Dynamo bytecode compilation once,
    and run Inductor compilation multiple times with different shapes
    plus a general shape. The compilation for specific shapes happens
    outside of the context of the Dynamo bytecode compilation. At that
    time, we don't have shape environment to provide to Inductor, and
    it will fail the Inductor code cache lookup.

    By providing a dummy shape environment that always hits, we can
    make the Inductor code cache lookup always hit, and we can
    compile the graph for different shapes as needed.

    The following dummy methods are obtained by trial-and-error
    until it works.
    """

    def __init__(self) -> None:
        self.guards = []

    def evaluate_guards_expression(self, *args, **kwargs):
        return True

    def get_pruned_guards(self, *args, **kwargs):
        return []

    def produce_guards_expression(self, *args, **kwargs):
        return ""

# =============== tests ===================

mod = torch.nn.Linear(1, 3)
x = torch.randn(4, 1)
torch._dynamo.mark_dynamic(x, 0)

def f(x):
    with torch.no_grad():
        return mod(x)

gm, args, kwargs = capture(f)(x)
assert not kwargs

# Problem 1: args is a SymInt, which needs a shape_env.
# How do we pass that to inductor_compile?
compiled_artifact = inductor_compile(gm, args)
print("inductor compile")
compiled_artifact.save("compiled_artifact.bin")
print("saved artifact")
loaded = CompiledArtifact.load("compiled_artifact.bin", args)
print("loaded artifact")
out = loaded(*args)
