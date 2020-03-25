import torch


class TorchTensorEngine(object):
    def rand(self, shape, device=None, requires_grad=False):
        return torch.rand(shape, device=device, requires_grad=requires_grad)

    def nchw_rand(self, shape, device=None, requires_grad=False):
        return self.rand(shape, device=device, requires_grad=requires_grad)

    def reset(self, _):
        pass

    def rand_like(self, v):
        return torch.rand_like(v)

    def numpy(self, t):
        return t.cpu().numpy()

    def mul(self, t1, t2):
        return t1 * t2

    def add(self, t1, t2):
        return t1 + t2

    def batch_norm(self, data, mean, var, training):
        return torch.nn.functional.batch_norm(data, mean, var, training=training)

    def instance_norm(self, data):
        return torch.nn.functional.instance_norm(data)

    def layer_norm(self, data, shape):
        return torch.nn.functional.layer_norm(data, shape)

    def sync_cuda(self):
        torch.cuda.synchronize()

    def backward(self, tensors, grad_tensors, _):
        torch.autograd.backward(tensors, grad_tensors=grad_tensors)

    def sum(self, data, dims):
        return torch.sum(data, dims)

    def softmax(self, data, dim=None):
        return torch.nn.functional.softmax(data, dim)

    def max_pool2d(self, data, kernel_size, stride=1):
        return torch.nn.functional.max_pool2d(data, kernel_size, stride=stride)

    def avg_pool2d(self, data, kernel_size, stride=1):
        return torch.nn.functional.avg_pool2d(data, kernel_size, stride=stride)

    def conv2d_layer(self, ic, oc, kernel_size, groups=1):
        return torch.nn.Conv2d(ic, oc, kernel_size, groups=groups)

    def matmul(self, t1, t2):
        return torch.matmul(t1, t2)

    def to_device(self, module, device):
        return module.to(device)
