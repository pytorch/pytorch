import torch
from torch.utils._pytree import tree_map
from torch.fx.passes.backends.cudagraphs import partition_cudagraphs
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
try:
    import torchdynamo
    TEST_DYNAMO = True
except ImportError:
    TEST_DYNAMO = False

TEST_CUDA = torch.cuda.is_available()

if not TEST_CUDA or not TEST_DYNAMO:
    print('CUDA or dynamo not available, skipping tests', file=sys.stderr)
    TestCase = object  # noqa: F811

torchdynamo.config.verify_correctness = True

class RunCudaGraph():
    warmed_up = False

    # these are all None or all filled
    graph = None
    static_inputs = None
    static_outputs = None

    def __call__(self, gm, *args):
        def cloner(t):
            if isinstance(t, torch.Tensor):
                return t.clone()
            else:
                return t

        # TODO: once we've recorded here, we'd like to replace this with
        # compiled bytecode that copies into static, replays the cuda graph,
        # then copies out.  First condition is the hotpath, needs optimizing
        if self.graph is not None:
            assert len(args) == len(self.static_inputs)
            for dst, src in zip(self.static_inputs, args):
                dst.copy_(src)
            self.graph.replay()
            return tree_map(cloner, self.static_outputs)

        elif self.warmed_up:
            # record
            self.static_inputs = [x.clone() for x in args]
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # TODO: this is wrong but required due to _wrapped_call
                self.static_outputs = super(type(gm), gm).__call__(*self.static_inputs)
            return tree_map(cloner, self.static_outputs)

        else:
            # warmup
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # TODO: this is wrong but required due to _wrapped_call
                r = super(type(gm), gm).__call__(*args)
            torch.cuda.current_stream().wait_stream(stream)
            self.warmed_up = True
            return r

def cudagraphs(model, inputs):
    model = partition_cudagraphs(model, inputs)

    # Do some swizzling
    for node in model.graph.nodes:
        # TODO: this is wrong, cribbed from
        # https://github.com/pytorch/pytorch/pull/80591
        if "fused_" in node.name:
            fused_module = getattr(model, node.name)
            # TODO: this is also wrong
            fused_module._wrapped_call = RunCudaGraph()

    # model here is compiled FX graph, so it should be reasonably efficient
    return model


class TestDynamoCudaGraphs(TestCase):
    def test_basic(self):
        def model(x, y):
            return (x + y) * y

        results = []
        with torchdynamo.optimize(cudagraphs):
            for i in range(5):
                x = torch.randn(3, device='cuda')
                y = torch.randn(3, device='cuda')
                results.append(model(x, y))

    def test_dtoh(self):
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b

        results = []
        with torchdynamo.optimize(cudagraphs):
            for i in range(5):
                x = torch.randn(3, device='cuda')
                y = torch.randn(3, device='cuda')
                results.append(model(x, y))

    def test_htod(self):
        def model(x, y):
            a = x + y
            return a * 3

        results = []
        with torchdynamo.optimize(cudagraphs):
            for i in range(5):
                x = torch.randn(3, device='cuda')
                y = torch.randn((), device='cpu')
                results.append(model(x, y))

if __name__ == "__main__":
    run_tests()
