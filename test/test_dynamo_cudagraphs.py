import torch
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

def cudagraphs(model, inputs):
    assert isinstance(inputs, (list, tuple))
    static_inputs = [torch.zeros_like(x) for x in inputs]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        graph.replay()
        return [x.clone() for x in static_outputs]

    return run


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
