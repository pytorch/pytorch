import torch
def test_foreach_copy_h2d():
    self_tensor = [torch.randn(1,2), torch.randn(1,3)]
    src_tensor = [torch.randn(1,2), torch.randn(1,3)]
    def fn(h1, h2):
        return torch.ops.aten._foreach_copy(h1, h2)
    cpu_result = fn(self_tensor, src_tensor)

    fn = torch.compile(fn)
    # self_tensor : cuda
    # src_tensor : cpu 
    for i in range(len(self_tensor)):
        self_tensor[i] = self_tensor[i].to("cuda")
    cuda_result = fn(self_tensor, src_tensor)
    for i in range (len(self_tensor)):
        assert torch.allclose(cpu_result[i], cuda_result[i].cpu(), rtol=1e-4, atol=1e-4)

def test_foreach_copy_d2h():
    self_tensor = [torch.randn(1,2), torch.randn(1,3)]
    src_tensor = [torch.randn(1,2), torch.randn(1,3)]
    def fn(h1, h2):
        return torch.ops.aten._foreach_copy(h1, h2)
    cpu_result = fn(self_tensor, src_tensor)

    fn = torch.compile(fn)
    # self_tensor : cpu
    # src_tensor : cuda
    for i in range(len(self_tensor)):
        src_tensor[i] = src_tensor[i].to("cuda")
    cuda_result = fn(self_tensor, src_tensor)
    for i in range (len(self_tensor)):
        assert torch.allclose(cpu_result[i], cuda_result[i], rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
  test_foreach_copy_h2d()
  test_foreach_copy_d2h()