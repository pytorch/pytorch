import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests, OpDTypes, onlyCUDA, onlyCPU
from torch.testing._internal.common_utils import TestCase, run_tests

class TestCommon(TestCase):

    @ops([op for op in op_db if op.supports_out], allowed_dtypes=(torch.float32,))
    def test_dynamo_out(self, device, dtype, op):
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True

        fullgraph = False
        backend = "eager"

        samples = list(op.sample_inputs(device, dtype))

        for i, sample in enumerate(samples):
            torch._dynamo.reset()
            input, args, kwargs = (sample.input, sample.args, sample.kwargs)

            # Run the functional version of the operation, using eager.
            try:
                expected = op(input, *args, **kwargs)

                if isinstance(expected, tuple):
                    expected = tuple(expected)
            except:
                # If that doesn't work out, go to the next sample.
                continue

            def op_out(input, args, kwargs, expected):
                # Create the output inside the compiled function, since resizing
                # graph inputs are not allowed.
                out = pytree.tree_map_only(torch.Tensor, lambda t: torch.empty_like(t), expected)
                return op(input, *args, **kwargs, out=out)

            def run(f):
                # Try running the operation, and return the raised error, if any.
                try:
                    f(input, args, kwargs, expected)
                except Exception as e:
                    return e

            eager_err = run(op_out)
            dynamo_err = run(torch.compile(op_out, backend=backend, fullgraph=fullgraph))

            if eager_err is None and dynamo_err is not None:
                raise RuntimeError(f"eager didn't fail, but dynamo did.") from dynamo_err
            elif eager_err is not None and dynamo_err is None:
                raise RuntimeError(f"eager failed, but dynamo didn't.") from eager_err

instantiate_device_type_tests(TestCommon, globals())

if __name__ == "__main__":
    run_tests()
