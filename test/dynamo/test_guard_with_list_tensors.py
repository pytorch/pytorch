import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._C._dynamo import guards

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@torch.compile
def fn(a, b, data):
    x = a + b
    if data is not None:
        x = x + data[0] + data[1]
    return x

class TestGuardWithListofTensors(torch._dynamo.test_case.TestCase):
    def test_tensor_duplicated_outbound(self):
        t1, t2 = torch.randn(4), torch.randn(4)
        fn(t1, t2, [torch.randn(4), torch.randn(4)])
        t = torch.randn(4)
        fn(t, t, None)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
