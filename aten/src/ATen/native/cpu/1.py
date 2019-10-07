import torch
import torch.nn.functional as F
import math  # type: ignore
import numpy  # type: ignore
import io  # type: ignore
import itertools  # type: ignore

def preprocess(inp):
    # type: (torch.Tensor) -> torch.Tensor
    return inp


def example_torch_Generator():
    g_cpu = torch.Generator()
    g_cuda = torch.Generator(device='cuda')


def example_torch_abs():
    torch.abs(torch.tensor([-1, -2, 3]))


def example_torch_acos():
    a = torch.randn(4)
    a
    torch.acos(a)


def example_torch_add():
    a = torch.randn(4)
    a
    torch.add(a, 20)
    a = torch.randn(4)
    a
    b = torch.randn(4, 1)
    b
    torch.add(a, 10, b)


def example_torch_addbmm():
    M = torch.randn(3, 5)
    batch1 = torch.randn(10, 3, 4)
    batch2 = torch.randn(10, 4, 5)
    torch.addbmm(M, batch1, batch2)


def example_torch_addcdiv():
    t = torch.randn(1, 3)
    t1 = torch.randn(3, 1)
    t2 = torch.randn(1, 3)
    torch.addcdiv(t, 0.1, t1, t2)


def example_torch_addcmul():
    t = torch.randn(1, 3)
    t1 = torch.randn(3, 1)
    t2 = torch.randn(1, 3)
    torch.addcmul(t, 0.1, t1, t2)


def example_torch_addmm():
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    torch.addmm(M, mat1, mat2)


def example_torch_addmv():
    M = torch.randn(2)
    mat = torch.randn(2, 3)
    vec = torch.randn(3)
    torch.addmv(M, mat, vec)


def example_torch_addr():
    vec1 = torch.arange(1., 4.)
    vec2 = torch.arange(1., 3.)
    M = torch.zeros(3, 2)
    torch.addr(M, vec1, vec2)


def example_torch_allclose():
    torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
    torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
    torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
    torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)


def example_torch_arange():
    torch.arange(5)
    torch.arange(1, 4)
    torch.arange(1, 2.5, 0.5)


def example_torch_argmax():
    a = torch.randn(4, 4)
    a
    torch.argmax(a)
    a = torch.randn(4, 4)
    a
    torch.argmax(a, dim=1)


def example_torch_argmin():
    a = torch.randn(4, 4)
    a
    torch.argmin(a)
    a = torch.randn(4, 4)
    a
    torch.argmin(a, dim=1)


def example_torch_argsort():
    a = torch.randn(4, 4)
    a
    torch.argsort(a, dim=1)


def example_torch_as_strided():
    x = torch.randn(3, 3)
    x
    t = torch.as_strided(x, (2, 2), (1, 2))
    t
    t = torch.as_strided(x, (2, 2), (1, 2), 1)


def example_torch_as_tensor():
    a = numpy.array([1, 2, 3])
    t = torch.as_tensor(a)
    t
    t[0] = -1
    a
    a = numpy.array([1, 2, 3])
    t = torch.as_tensor(a, device=torch.device('cuda'))
    t
    t[0] = -1
    a


def example_torch_asin():
    a = torch.randn(4)
    a
    torch.asin(a)


def example_torch_atan():
    a = torch.randn(4)
    a
    torch.atan(a)


def example_torch_atan2():
    a = torch.randn(4)
    a
    torch.atan2(a, torch.randn(4))


def example_torch_avg_pool1d():
    # pool of square window of size=3, stride=2
    input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
    F.avg_pool1d(input, kernel_size=3, stride=2)


def example_torch_baddbmm():
    M = torch.randn(10, 3, 5)
    batch1 = torch.randn(10, 3, 4)
    batch2 = torch.randn(10, 4, 5)
    torch.baddbmm(M, batch1, batch2).size()


def example_torch_bernoulli():
    a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
    a
    torch.bernoulli(a)
    a = torch.ones(3, 3) # probability of drawing "1" is 1
    torch.bernoulli(a)
    a = torch.zeros(3, 3) # probability of drawing "1" is 0
    torch.bernoulli(a)


def example_torch_bincount():
    input = torch.randint(0, 8, (5,), dtype=torch.int64)
    weights = torch.linspace(0, 1, steps=5)
    input, weights
    torch.bincount(input)
    input.bincount(weights)


def example_torch_bitwise_not():
    torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))


def example_torch_bmm():
    input = torch.randn(10, 3, 4)
    mat2 = torch.randn(10, 4, 5)
    res = torch.bmm(input, mat2)
    res.size()


def example_torch_broadcast_tensors():
    x = torch.arange(3).view(1, 3)
    y = torch.arange(2).view(2, 1)
    a, b = torch.broadcast_tensors(x, y)
    a.size()
    a


def example_torch_can_cast():
    torch.can_cast(torch.double, torch.float)
    torch.can_cast(torch.float, torch.int)


def example_torch_cartesian_prod():
    a = [1, 2, 3]
    b = [4, 5]
    list(itertools.product(a, b))
    tensor_a = torch.tensor(a)
    tensor_b = torch.tensor(b)
    torch.cartesian_prod(tensor_a, tensor_b)


def example_torch_cat():
    x = torch.randn(2, 3)
    x
    torch.cat((x, x, x), 0)
    torch.cat((x, x, x), 1)


def example_torch_cdist():
    a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
    a
    b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
    b
    torch.cdist(a, b, p=2)


def example_torch_ceil():
    a = torch.randn(4)
    a
    torch.ceil(a)


def example_torch_chain_matmul():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    c = torch.randn(5, 6)
    d = torch.randn(6, 7)
    torch.chain_matmul(a, b, c, d)


def example_torch_cholesky():
    a = torch.randn(3, 3)
    a = torch.mm(a, a.t()) # make symmetric positive-definite
    l = torch.cholesky(a)
    a
    l
    torch.mm(l, l.t())
    a = torch.randn(3, 2, 2)
    a = torch.matmul(a, a.transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
    l = torch.cholesky(a)
    z = torch.matmul(l, l.transpose(-1, -2))
    torch.max(torch.abs(z - a)) # Max non-zero


def example_torch_cholesky_inverse():
    a = torch.randn(3, 3)
    a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positivedefinite
    u = torch.cholesky(a)
    a
    torch.cholesky_inverse(u)
    a.inverse()


def example_torch_cholesky_solve():
    a = torch.randn(3, 3)
    a = torch.mm(a, a.t()) # make symmetric positivedefinite
    u = torch.cholesky(a)
    a
    b = torch.randn(3, 2)
    b
    torch.cholesky_solve(b, u)
    torch.mm(a.inverse(), b)


def example_torch_clamp():
    a = torch.randn(4)
    a
    torch.clamp(a, min=-0.5, max=0.5)
    a = torch.randn(4)
    a
    torch.clamp(a, min=0.5)
    a = torch.randn(4)
    a
    torch.clamp(a, max=0.5)


def example_torch_combinations():
    a = [1, 2, 3]
    list(itertools.combinations(a, r=2))
    list(itertools.combinations(a, r=3))
    list(itertools.combinations_with_replacement(a, r=2))
    tensor_a = torch.tensor(a)
    torch.combinations(tensor_a)
    torch.combinations(tensor_a, r=3)
    torch.combinations(tensor_a, with_replacement=True)


def example_torch_conv1d():
    filters = torch.randn(33, 16, 3)
    inputs = torch.randn(20, 16, 50)
    F.conv1d(inputs, filters)


def example_torch_conv2d():
    # With square kernels and equal stride
    filters = torch.randn(8,4,3,3)
    inputs = torch.randn(1,4,5,5)
    F.conv2d(inputs, filters, padding=1)


def example_torch_conv3d():
    filters = torch.randn(33, 16, 3, 3, 3)
    inputs = torch.randn(20, 16, 50, 10, 20)
    F.conv3d(inputs, filters)


def example_torch_conv_transpose1d():
    inputs = torch.randn(20, 16, 50)
    weights = torch.randn(16, 33, 5)
    F.conv_transpose1d(inputs, weights)


def example_torch_conv_transpose2d():
    # With square kernels and equal stride
    inputs = torch.randn(1, 4, 5, 5)
    weights = torch.randn(4, 8, 3, 3)
    F.conv_transpose2d(inputs, weights, padding=1)


def example_torch_conv_transpose3d():
    inputs = torch.randn(20, 16, 50, 10, 20)
    weights = torch.randn(16, 33, 3, 3, 3)
    F.conv_transpose3d(inputs, weights)


def example_torch_cos():
    a = torch.randn(4)
    a
    torch.cos(a)


def example_torch_cosh():
    a = torch.randn(4)
    a
    torch.cosh(a)


def example_torch_cosine_similarity():
    input1 = torch.randn(100, 128)
    input2 = torch.randn(100, 128)
    output = F.cosine_similarity(input1, input2)
    print(output)


def example_torch_cross():
    a = torch.randn(4, 3)
    a
    b = torch.randn(4, 3)
    b
    torch.cross(a, b, dim=1)
    torch.cross(a, b)


def example_torch_cumprod():
    a = torch.randn(10)
    a
    torch.cumprod(a, dim=0)
    a[5] = 0.0
    torch.cumprod(a, dim=0)


def example_torch_cumsum():
    a = torch.randn(10)
    a
    torch.cumsum(a, dim=0)


def example_torch_det():
    A = torch.randn(3, 3)
    torch.det(A)
    A = torch.randn(3, 2, 2)
    A
    A.det()


def example_torch_diag():
    a = torch.randn(3)
    a
    torch.diag(a)
    torch.diag(a, 1)
    a = torch.randn(3, 3)
    a
    torch.diag(a, 0)
    torch.diag(a, 1)


def example_torch_diag_embed():
    a = torch.randn(2, 3)
    torch.diag_embed(a)
    torch.diag_embed(a, offset=1, dim1=0, dim2=2)


def example_torch_diagflat():
    a = torch.randn(3)
    a
    torch.diagflat(a)
    torch.diagflat(a, 1)
    a = torch.randn(2, 2)
    a
    torch.diagflat(a)


def example_torch_diagonal():
    a = torch.randn(3, 3)
    a
    torch.diagonal(a, 0)
    torch.diagonal(a, 1)
    x = torch.randn(2, 5, 4, 2)
    torch.diagonal(x, offset=-1, dim1=1, dim2=2)


def example_torch_digamma():
    a = torch.tensor([1, 0.5])
    torch.digamma(a)


def example_torch_dist():
    x = torch.randn(4)
    x
    y = torch.randn(4)
    y
    torch.dist(x, y, 3.5)
    torch.dist(x, y, 3)
    torch.dist(x, y, 0)
    torch.dist(x, y, 1)


def example_torch_div():
    a = torch.randn(5)
    a
    torch.div(a, 0.5)
    a = torch.randn(4, 4)
    a
    b = torch.randn(4)
    b
    torch.div(a, b)


def example_torch_dot():
    torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))


def example_torch_einsum():
    x = torch.randn(5)
    y = torch.randn(4)
    torch.einsum('i,j->ij', x, y)  # outer product
    A = torch.randn(3,5,4)
    l = torch.randn(2,5)
    r = torch.randn(2,4)
    torch.einsum('bn,anm,bm->ba', l, A, r) # compare torch.nn.functional.bilinear
    As = torch.randn(3,2,5)
    Bs = torch.randn(3,5,4)
    torch.einsum('bij,bjk->bik', As, Bs) # batch matrix multiplication
    A = torch.randn(3, 3)
    torch.einsum('ii->i', A) # diagonal
    A = torch.randn(4, 3, 3)
    torch.einsum('...ii->...i', A) # batch diagonal
    A = torch.randn(2, 3, 4, 5)
    torch.einsum('...ij->...ji', A).shape # batch permute


def example_torch_empty():
    torch.empty(2, 3)


def example_torch_empty_like():
    torch.empty((2,3), dtype=torch.int64)


def example_torch_empty_strided():
    a = torch.empty_strided((2, 3), (1, 2))
    a
    a.stride()
    a.size()


def example_torch_enable_grad():
    x = torch.tensor([1], requires_grad=True)
    with torch.no_grad():
        with torch.enable_grad():
            y = x * 2
    y.requires_grad
    y.backward()
    x.grad
    @torch.enable_grad()
def doubler(x):
    return x * 2
with torch.no_grad():
    z = doubler(x)
z.requires_grad


def example_torch_eq():
    torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))


def example_torch_equal():
    torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))


def example_torch_erf():
    torch.erf(torch.tensor([0, -1., 10.]))


def example_torch_erfc():
    torch.erfc(torch.tensor([0, -1., 10.]))


def example_torch_erfinv():
    torch.erfinv(torch.tensor([0, 0.5, -1.]))


def example_torch_exp():
    torch.exp(torch.tensor([0, math.log(2.)]))


def example_torch_expm1():
    torch.expm1(torch.tensor([0, math.log(2.)]))


def example_torch_eye():
    torch.eye(3)


def example_torch_fft():
    # unbatched 2D FFT
    x = torch.randn(4, 3, 2)
    torch.fft(x, 2)
    # batched 1D FFT
    torch.fft(x, 1)
    # arbitrary number of batch dimensions, 2D FFT
    x = torch.randn(3, 3, 5, 5, 2)
    y = torch.fft(x, 2)
    y.shape


def example_torch_flatten():
    t = torch.tensor([[[1, 2],
                       [3, 4]],
                      [[5, 6],
                       [7, 8]]])
    torch.flatten(t)
    torch.flatten(t, start_dim=1)


def example_torch_flip():
    x = torch.arange(8).view(2, 2, 2)
    x
    torch.flip(x, [0, 1])


def example_torch_floor():
    a = torch.randn(4)
    a
    torch.floor(a)


def example_torch_fmod():
    torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    torch.fmod(torch.tensor([1., 2, 3, 4, 5]), 1.5)


def example_torch_frac():
    torch.frac(torch.tensor([1, 2.5, -3.2]))


def example_torch_from_numpy():
    a = numpy.array([1, 2, 3])
    t = torch.from_numpy(a)
    t
    t[0] = -1
    a


def example_torch_full():
    torch.full((2, 3), 3.141592)


def example_torch_gather():
    t = torch.tensor([[1,2],[3,4]])
    torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))


def example_torch_ge():
    torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))


def example_torch_ger():
    v1 = torch.arange(1., 5.)
    v2 = torch.arange(1., 4.)
    torch.ger(v1, v2)


def example_torch_get_default_dtype():
    torch.get_default_dtype()  # initialdefault for floating point is torch.float32
    torch.set_default_dtype(torch.float64)
    torch.get_default_dtype()  #default is now changed to torch.float64
    torch.set_default_tensor_type(torch.FloatTensor)  # setting tensor type also affects this
    torch.get_default_dtype()  # changed to torch.float32, the dtype for torch.FloatTensor


def example_torch_gt():
    torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))


def example_torch_histc():
    torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)


def example_torch_ifft():
    x = torch.randn(3, 3, 2)
    x
    y = torch.fft(x, 2)
    torch.ifft(y, 2)  # recover x


def example_torch_index_select():
    x = torch.randn(3, 4)
    x
    indices = torch.tensor([0, 2])
    torch.index_select(x, 0, indices)
    torch.index_select(x, 1, indices)


def example_torch_inverse():
    x = torch.rand(4, 4)
    y = torch.inverse(x)
    z = torch.mm(x, y)
    z
    torch.max(torch.abs(z - torch.eye(4))) # Max non-zero
    # Batched inverse example
    x = torch.randn(2, 3, 4, 4)
    y = torch.inverse(x)
    z = torch.matmul(x, y)
    torch.max(torch.abs(z - torch.eye(4).expand_as(x))) # Max non-zero


def example_torch_irfft():
    x = torch.randn(4, 4)
    torch.rfft(x, 2, onesided=True).shape
    # notice that with onesided=True, output size does not determine the original signal size
    x = torch.randn(4, 5)
    torch.rfft(x, 2, onesided=True).shape
    # now we use the original shape to recover x
    x
    y = torch.rfft(x, 2, onesided=True)
    torch.irfft(y, 2, onesided=True, signal_sizes=x.shape)  # recover x


def example_torch_isfinite():
    torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))


def example_torch_isinf():
    torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))


def example_torch_isnan():
    torch.isnan(torch.tensor([1, float('nan'), 2]))


def example_torch_kthvalue():
    x = torch.arange(1., 6.)
    x
    torch.kthvalue(x, 4)
    x=torch.arange(1.,7.).resize_(2,3)
    x
    torch.kthvalue(x, 2, 0, True)


def example_torch_le():
    torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))


def example_torch_lerp():
    start = torch.arange(1., 5.)
    end = torch.empty(4).fill_(10)
    start
    end
    torch.lerp(start, end, 0.5)
    torch.lerp(start, end, torch.full_like(start, 0.5))


def example_torch_linspace():
    torch.linspace(3, 10, steps=5)
    torch.linspace(-10, 10, steps=5)
    torch.linspace(start=-10, end=10, steps=5)
    torch.linspace(start=-10, end=10, steps=1)


def example_torch_load():
    torch.load('tensors.pt')
    torch.load('tensors.pt', map_location=torch.device('cpu'))
    torch.load('tensors.pt', map_location=lambda storage, loc: storage)
    torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
    torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
    with open('tensor.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    torch.load(buffer)
    torch.load('module.pt', encoding='ascii')


def example_torch_log():
    a = torch.randn(5)
    a
    torch.log(a)


def example_torch_log10():
    a = torch.rand(5)
    a
    torch.log10(a)


def example_torch_log1p():
    a = torch.randn(5)
    a
    torch.log1p(a)


def example_torch_log2():
    a = torch.rand(5)
    a
    torch.log2(a)


def example_torch_logdet():
    A = torch.randn(3, 3)
    torch.det(A)
    torch.logdet(A)
    A
    A.det()
    A.det().log()


def example_torch_logical_not():
    torch.logical_not(torch.tensor([True, False]))
    torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8))
    torch.logical_not(torch.tensor([0., 1.5, -10.], dtype=torch.double))
    torch.logical_not(torch.tensor([0., 1., -10.], dtype=torch.double), out=torch.empty(3, dtype=torch.int16))


def example_torch_logical_xor():
    torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))


def example_torch_logspace():
    torch.logspace(start=-10, end=10, steps=5)
    torch.logspace(start=0.1, end=1.0, steps=5)
    torch.logspace(start=0.1, end=1.0, steps=1)
    torch.logspace(start=2, end=2, steps=1, base=2)


def example_torch_logsumexp():
    a = torch.randn(3, 3)
    torch.logsumexp(a, 1)


def example_torch_lstsq():
    A = torch.tensor([[1., 1, 1],
                      [2, 3, 4],
                      [3, 5, 2],
                      [4, 2, 5],
                      [5, 4, 3]])
    B = torch.tensor([[-10., -3],
                      [ 12, 14],
                      [ 14, 12],
                      [ 16, 16],
                      [ 18, 16]])
    X, _ = torch.lstsq(B, A)
    X


def example_torch_lt():
    torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))


def example_torch_lu():
    A = torch.randn(2, 3, 3)
    A_LU, pivots = torch.lu(A)
    A_LU
    pivots
    A_LU, pivots, info = torch.lu(A, get_infos=True)
    if info.nonzero().size(0) == 0:
        print('LU factorization succeeded for all samples!')


def example_torch_lu_solve():
    A = torch.randn(2, 3, 3)
    b = torch.randn(2, 3, 1)
    A_LU = torch.lu(A)
    x = torch.lu_solve(b, *A_LU)
    torch.norm(torch.bmm(A, x) - b)


def example_torch_lu_unpack():
    A = torch.randn(2, 3, 3)
    A_LU, pivots = A.lu()
    P, A_L, A_U = torch.lu_unpack(A_LU, pivots)
    # can recover A from factorization
    A_ = torch.bmm(P, torch.bmm(A_L, A_U))


def example_torch_masked_select():
    x = torch.randn(3, 4)
    x
    mask = x.ge(0.5)
    mask
    torch.masked_select(x, mask)


def example_torch_matmul():
    # vector x vector
    tensor1 = torch.randn(3)
    tensor2 = torch.randn(3)
    torch.matmul(tensor1, tensor2).size()
    # matrix x vector
    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(4)
    torch.matmul(tensor1, tensor2).size()
    # batched matrix x broadcasted vector
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(4)
    torch.matmul(tensor1, tensor2).size()
    # batched matrix x batched matrix
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(10, 4, 5)
    torch.matmul(tensor1, tensor2).size()
    # batched matrix x broadcasted matrix
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(4, 5)
    torch.matmul(tensor1, tensor2).size()


def example_torch_matrix_power():
    a = torch.randn(2, 2, 2)
    a
    torch.matrix_power(a, 3)


def example_torch_matrix_rank():
    a = torch.eye(10)
    torch.matrix_rank(a)
    b = torch.eye(10)
    b[0, 0] = 0
    torch.matrix_rank(b)


def example_torch_max():
    a = torch.randn(1, 3)
    a
    torch.max(a)
    a = torch.randn(4, 4)
    a
    torch.max(a, 1)
    a = torch.randn(4)
    a
    b = torch.randn(4)
    b
    torch.max(a, b)


def example_torch_mean():
    a = torch.randn(1, 3)
    a
    torch.mean(a)
    a = torch.randn(4, 4)
    a
    torch.mean(a, 1)
    torch.mean(a, 1, True)


def example_torch_median():
    a = torch.randn(1, 3)
    a
    torch.median(a)
    a = torch.randn(4, 5)
    a
    torch.median(a, 1)


def example_torch_meshgrid():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_x
    grid_y


def example_torch_min():
    a = torch.randn(1, 3)
    a
    torch.min(a)
    a = torch.randn(4, 4)
    a
    torch.min(a, 1)
    a = torch.randn(4)
    a
    b = torch.randn(4)
    b
    torch.min(a, b)


def example_torch_mm():
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    torch.mm(mat1, mat2)


def example_torch_mode():
    a = torch.randint(10, (5,))
    a
    b = a + (torch.randn(50, 1) * 5).long()
    torch.mode(b, 0)


def example_torch_mul():
    a = torch.randn(3)
    a
    torch.mul(a, 100)
    a = torch.randn(4, 1)
    a
    b = torch.randn(1, 4)
    b
    torch.mul(a, b)


def example_torch_multinomial():
    weights = torch.tensor([0, 10, 3, 0], dtype=torch.float) # create a tensor of weights
    torch.multinomial(weights, 2)
    torch.multinomial(weights, 4) # ERROR!
    torch.multinomial(weights, 4, replacement=True)


def example_torch_mv():
    mat = torch.randn(2, 3)
    vec = torch.randn(3)
    torch.mv(mat, vec)


def example_torch_mvlgamma():
    a = torch.empty(2, 3).uniform_(1, 2)
    a
    torch.mvlgamma(a, 2)


def example_torch_narrow():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    torch.narrow(x, 0, 0, 2)
    torch.narrow(x, 1, 1, 2)


def example_torch_ne():
    torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))


def example_torch_neg():
    a = torch.randn(5)
    a
    torch.neg(a)


def example_torch_no_grad():
    x = torch.tensor([1], requires_grad=True)
    with torch.no_grad():
        y = x * 2
    y.requires_grad
    @torch.no_grad()
def doubler(x):
    return x * 2
z = doubler(x)
z.requires_grad


def example_torch_nonzero():
    torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
    torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                [0.0, 0.4, 0.0, 0.0],
                                [0.0, 0.0, 1.2, 0.0],
                                [0.0, 0.0, 0.0,-0.4]]))
    torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
    torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                [0.0, 0.4, 0.0, 0.0],
                                [0.0, 0.0, 1.2, 0.0],
                                [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
    torch.nonzero(torch.tensor(5), as_tuple=True)


def example_torch_norm():
    import torch
    a = torch.arange(9, dtype= torch.float) - 4
    b = a.reshape((3, 3))
    torch.norm(a)
    torch.norm(b)
    torch.norm(a, float('inf'))
    torch.norm(b, float('inf'))
    c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
    torch.norm(c, dim=0)
    torch.norm(c, dim=1)
    torch.norm(c, p=1, dim=1)
    d = torch.arange(8, dtype= torch.float).reshape(2,2,2)
    torch.norm(d, dim=(1,2))
    torch.norm(d[0, :, :]), torch.norm(d[1, :, :])


def example_torch_normal():
    torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
    torch.normal(mean=0.5, std=torch.arange(1., 6.))
    torch.normal(mean=torch.arange(1., 6.))
    torch.normal(2, 3, size=(1, 4))


def example_torch_numel():
    a = torch.randn(1, 2, 3, 4, 5)
    torch.numel(a)
    a = torch.zeros(4,4)
    torch.numel(a)


def example_torch_ones():
    torch.ones(2, 3)
    torch.ones(5)


def example_torch_ones_like():
    input = torch.empty(2, 3)
    torch.ones_like(input)


def example_torch_pinverse():
    input = torch.randn(3, 5)
    input
    torch.pinverse(input)


def example_torch_pixel_shuffle():
    input = torch.randn(1, 9, 4, 4)
    output = torch.nn.functional.pixel_shuffle(input, 3)
    print(output.size())


def example_torch_pow():
    a = torch.randn(4)
    a
    torch.pow(a, 2)
    exp = torch.arange(1., 5.)
    a = torch.arange(1., 5.)
    a
    exp
    torch.pow(a, exp)
    exp = torch.arange(1., 5.)
    base = 2
    torch.pow(base, exp)


def example_torch_prod():
    a = torch.randn(1, 3)
    a
    torch.prod(a)
    a = torch.randn(4, 2)
    a
    torch.prod(a, 1)


def example_torch_qr():
    a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    q, r = torch.qr(a)
    q
    r
    torch.mm(q, r).round()
    torch.mm(q.t(), q).round()
    a = torch.randn(3, 4, 5)
    q, r = torch.qr(a, some=False)
    torch.allclose(torch.matmul(q, r), a)
    torch.allclose(torch.matmul(q.transpose(-2, -1), q), torch.eye(5))


def example_torch_rand():
    torch.rand(4)
    torch.rand(2, 3)


def example_torch_randint():
    torch.randint(3, 5, (3,))
    torch.randint(10, (2, 2))
    torch.randint(3, 10, (2, 2))


def example_torch_randn():
    torch.randn(4)
    torch.randn(2, 3)


def example_torch_randperm():
    torch.randperm(4)


def example_torch_range():
    torch.range(1, 4)
    torch.range(1, 4, 0.5)


def example_torch_reciprocal():
    a = torch.randn(4)
    a
    torch.reciprocal(a)


def example_torch_remainder():
    torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    torch.remainder(torch.tensor([1., 2, 3, 4, 5]), 1.5)


def example_torch_renorm():
    x = torch.ones(3, 3)
    x[1].fill_(2)
    x[2].fill_(3)
    x
    torch.renorm(x, 1, 0, 5)


def example_torch_repeat_interleave():
    x = torch.tensor([1, 2, 3])
    x.repeat_interleave(2)
    y = torch.tensor([[1, 2], [3, 4]])
    torch.repeat_interleave(y, 2)
    torch.repeat_interleave(y, 3, dim=1)
    torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)


def example_torch_reshape():
    a = torch.arange(4.)
    torch.reshape(a, (2, 2))
    b = torch.tensor([[0, 1], [2, 3]])
    torch.reshape(b, (-1,))


def example_torch_result_type():
    torch.result_type(torch.tensor([1, 2], dtype=torch.int), 1.0)
    torch.result_type(torch.tensor([1, 2], dtype=torch.uint8), torch.tensor(1))


def example_torch_rfft():
    x = torch.randn(5, 5)
    torch.rfft(x, 2).shape
    torch.rfft(x, 2, onesided=False).shape


def example_torch_roll():
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
    x
    torch.roll(x, 1, 0)
    torch.roll(x, -1, 0)
    torch.roll(x, shifts=(2, 1), dims=(0, 1))


def example_torch_rot90():
    x = torch.arange(4).view(2, 2)
    x
    torch.rot90(x, 1, [0, 1])
    x = torch.arange(8).view(2, 2, 2)
    x
    torch.rot90(x, 1, [1, 2])


def example_torch_round():
    a = torch.randn(4)
    a
    torch.round(a)


def example_torch_rsqrt():
    a = torch.randn(4)
    a
    torch.rsqrt(a)


def example_torch_save():
    # Save to file
    x = torch.tensor([0, 1, 2, 3, 4])
    torch.save(x, 'tensor.pt')
    # Save to io.BytesIO buffer
    buffer = io.BytesIO()
    torch.save(x, buffer)


def example_torch_set_default_dtype():
    torch.tensor([1.2, 3]).dtype           # initialdefault for floating point is torch.float32
    torch.set_default_dtype(torch.float64)
    torch.tensor([1.2, 3]).dtype           # a new floating point tensor


def example_torch_set_default_tensor_type():
    torch.tensor([1.2, 3]).dtype    # initialdefault for floating point is torch.float32
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.tensor([1.2, 3]).dtype    # a new floating point tensor


def example_torch_set_flush_denormal():
    torch.set_flush_denormal(True)
    torch.tensor([1e-323], dtype=torch.float64)
    torch.set_flush_denormal(False)
    torch.tensor([1e-323], dtype=torch.float64)


def example_torch_set_grad_enabled():
    x = torch.tensor([1], requires_grad=True)
    is_train = False
    with torch.set_grad_enabled(is_train):
        y = x * 2
    y.requires_grad
    torch.set_grad_enabled(True)
    y = x * 2
    y.requires_grad
    torch.set_grad_enabled(False)
    y = x * 2
    y.requires_grad


def example_torch_sigmoid():
    a = torch.randn(4)
    a
    torch.sigmoid(a)


def example_torch_sign():
    a = torch.tensor([0.7, -1.2, 0., 2.3])
    a
    torch.sign(a)


def example_torch_sin():
    a = torch.randn(4)
    a
    torch.sin(a)


def example_torch_sinh():
    a = torch.randn(4)
    a
    torch.sinh(a)


def example_torch_slogdet():
    A = torch.randn(3, 3)
    A
    torch.det(A)
    torch.logdet(A)
    torch.slogdet(A)


def example_torch_solve():
    A = torch.tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
                      [-6.05, -3.30,  5.36, -4.44,  1.08],
                      [-0.45,  2.58, -2.70,  0.27,  9.04],
                      [8.32,  2.71,  4.35,  -7.17,  2.14],
                      [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
    B = torch.tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
                      [-1.56,  4.00, -8.67,  1.75,  2.86],
                      [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
    X, LU = torch.solve(B, A)
    torch.dist(B, torch.mm(A, X))
    # Batched solver example
    A = torch.randn(2, 3, 1, 4, 4)
    B = torch.randn(2, 3, 1, 4, 6)
    X, LU = torch.solve(B, A)
    torch.dist(B, A.matmul(X))


def example_torch_sort():
    x = torch.randn(3, 4)
    sorted, indices = torch.sort(x)
    sorted
    indices
    sorted, indices = torch.sort(x, 0)
    sorted
    indices


def example_torch_sparse_coo_tensor():
    i = torch.tensor([[0, 1, 1],
                      [2, 0, 2]])
    v = torch.tensor([3, 4, 5], dtype=torch.float32)
    torch.sparse_coo_tensor(i, v, [2, 4])
    torch.sparse_coo_tensor(i, v)  # Shape inference
    torch.sparse_coo_tensor(i, v, [2, 4],
                            dtype=torch.float64,
                            device=torch.device('cuda:0'))
    S = torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
    S = torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])


def example_torch_sqrt():
    a = torch.randn(4)
    a
    torch.sqrt(a)


def example_torch_squeeze():
    x = torch.zeros(2, 1, 2, 1, 2)
    x.size()
    y = torch.squeeze(x)
    y.size()
    y = torch.squeeze(x, 0)
    y.size()
    y = torch.squeeze(x, 1)
    y.size()


def example_torch_std():
    a = torch.randn(1, 3)
    a
    torch.std(a)
    a = torch.randn(4, 4)
    a
    torch.std(a, dim=1)


def example_torch_std_mean():
    a = torch.randn(1, 3)
    a
    torch.std_mean(a)
    a = torch.randn(4, 4)
    a
    torch.std_mean(a, 1)


def example_torch_sum():
    a = torch.randn(1, 3)
    a
    torch.sum(a)
    a = torch.randn(4, 4)
    a
    torch.sum(a, 1)
    b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    torch.sum(b, (2, 1))


def example_torch_svd():
    a = torch.randn(5, 3)
    a
    u, s, v = torch.svd(a)
    u
    s
    v
    torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
    a_big = torch.randn(7, 5, 3)
    u, s, v = torch.svd(a_big)
    torch.dist(a_big, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1)))


def example_torch_symeig():
    a = torch.randn(5, 5)
    a = a + a.t()  # To make a symmetric
    a
    e, v = torch.symeig(a, eigenvectors=True)
    e
    v
    a_big = torch.randn(5, 2, 2)
    a_big = a_big + a_big.transpose(-2, -1)  # To make a_big symmetric
    e, v = a_big.symeig(eigenvectors=True)
    torch.allclose(torch.matmul(v, torch.matmul(e.diag_embed(), v.transpose(-2, -1))), a_big)


def example_torch_t():
    x = torch.randn(())
    x
    torch.t(x)
    x = torch.randn(3)
    x
    torch.t(x)
    x = torch.randn(2, 3)
    x
    torch.t(x)


def example_torch_take():
    src = torch.tensor([[4, 3, 5],
                        [6, 7, 8]])
    torch.take(src, torch.tensor([0, 2, 5]))


def example_torch_tan():
    a = torch.randn(4)
    a
    torch.tan(a)


def example_torch_tanh():
    a = torch.randn(4)
    a
    torch.tanh(a)


def example_torch_tensor():
    torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    torch.tensor([0, 1])  # Type inference on data
    torch.tensor([[0.11111, 0.222222, 0.3333333]],
                 dtype=torch.float64,
                 device=torch.device('cuda:0'))  # creates a torch.cuda.DoubleTensor
    torch.tensor(3.14159)  # Create a scalar (zero-dimensional tensor)
    torch.tensor([])  # Create an empty tensor (of size (0,))


def example_torch_tensordot():
    a = torch.arange(60.).reshape(3, 4, 5)
    b = torch.arange(24.).reshape(4, 3, 2)
    torch.tensordot(a, b, dims=([1, 0], [0, 1]))
    a = torch.randn(3, 4, 5, device='cuda')
    b = torch.randn(4, 5, 6, device='cuda')
    c = torch.tensordot(a, b, dims=2).cpu()


def example_torch_topk():
    x = torch.arange(1., 6.)
    x
    torch.topk(x, 3)


def example_torch_trace():
    x = torch.arange(1., 10.).view(3, 3)
    x
    torch.trace(x)


def example_torch_transpose():
    x = torch.randn(2, 3)
    x
    torch.transpose(x, 0, 1)


def example_torch_trapz():
    y = torch.randn((2, 3))
    y
    x = torch.tensor([[1, 3, 4], [1, 2, 3]])
    torch.trapz(y, x)


def example_torch_triangular_solve():
    A = torch.randn(2, 2).triu()
    A
    b = torch.randn(2, 3)
    b
    torch.triangular_solve(b, A)


def example_torch_tril():
    a = torch.randn(3, 3)
    a
    torch.tril(a)
    b = torch.randn(4, 6)
    b
    torch.tril(b, diagonal=1)
    torch.tril(b, diagonal=-1)


def example_torch_tril_indices():
    a = torch.tril_indices(3, 3)
    a
    a = torch.tril_indices(4, 3, -1)
    a
    a = torch.tril_indices(4, 3, 1)
    a


def example_torch_triu():
    a = torch.randn(3, 3)
    a
    torch.triu(a)
    torch.triu(a, diagonal=1)
    torch.triu(a, diagonal=-1)
    b = torch.randn(4, 6)
    b
    torch.triu(b, diagonal=1)
    torch.triu(b, diagonal=-1)


def example_torch_triu_indices():
    a = torch.triu_indices(3, 3)
    a
    a = torch.triu_indices(4, 3, -1)
    a
    a = torch.triu_indices(4, 3, 1)
    a


def example_torch_trunc():
    a = torch.randn(4)
    a
    torch.trunc(a)


def example_torch_unbind():
    torch.unbind(torch.tensor([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]]))


def example_torch_unique():
    output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
    output
    output, inverse_indices = torch.unique(
        torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True)
    output
    inverse_indices
    output, inverse_indices = torch.unique(
        torch.tensor([[1, 3], [2, 3]], dtype=torch.long), sorted=True, return_inverse=True)
    output
    inverse_indices


def example_torch_unique_consecutive():
    x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
    output = torch.unique_consecutive(x)
    output
    output, inverse_indices = torch.unique_consecutive(x, return_inverse=True)
    output
    inverse_indices
    output, counts = torch.unique_consecutive(x, return_counts=True)
    output
    counts


def example_torch_unsqueeze():
    x = torch.tensor([1, 2, 3, 4])
    torch.unsqueeze(x, 0)
    torch.unsqueeze(x, 1)


def example_torch_var():
    a = torch.randn(1, 3)
    a
    torch.var(a)
    a = torch.randn(4, 4)
    a
    torch.var(a, 1)


def example_torch_var_mean():
    a = torch.randn(1, 3)
    a
    torch.var_mean(a)
    a = torch.randn(4, 4)
    a
    torch.var_mean(a, 1)


def example_torch_where():
    x = torch.randn(3, 2)
    y = torch.ones(3, 2)
    x
    torch.where(x > 0, x, y)


def example_torch_zeros():
    torch.zeros(2, 3)
    torch.zeros(5)


def example_torch_zeros_like():
    input = torch.empty(2, 3)
    torch.zeros_like(input)


def example_torch_tensor_all():
    a = torch.rand(1, 2).bool()
    a
    a.all()
    a = torch.rand(4, 2).bool()
    a
    a.all(dim=1)
    a.all(dim=0)


def example_torch_tensor_any():
    a = torch.rand(1, 2).bool()
    a
    a.any()
    a = torch.randn(4, 2) < 0
    a
    a.any(1)
    a.any(0)


def example_torch_tensor_element_size():
    torch.tensor([]).element_size()
    torch.tensor([], dtype=torch.uint8).element_size()


def example_torch_tensor_expand():
    x = torch.tensor([[1], [2], [3]])
    x.size()
    x.expand(3, 4)
    x.expand(-1, 4)   # -1 means not changing the size of that dimension


def example_torch_tensor_fill_diagonal_():
    a = torch.zeros(3, 3)
    a.fill_diagonal_(5)
    b = torch.zeros(7, 3)
    b.fill_diagonal_(5)
    c = torch.zeros(7, 3)
    c.fill_diagonal_(5, wrap=True)


def example_torch_tensor_get_device():
    x = torch.randn(3, 4, 5, device='cuda:0')
    x.get_device()
    x.cpu().get_device()  # RuntimeError: get_device is not implemented for type torch.FloatTensor


def example_torch_tensor_index_add_():
    x = torch.ones(5, 3)
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    index = torch.tensor([0, 4, 2])
    x.index_add_(0, index, t)


def example_torch_tensor_index_copy_():
    x = torch.zeros(5, 3)
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    index = torch.tensor([0, 4, 2])
    x.index_copy_(0, index, t)


def example_torch_tensor_index_fill_():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    index = torch.tensor([0, 2])
    x.index_fill_(1, index, -1)


def example_torch_tensor_is_leaf():
    a = torch.rand(10, requires_grad=True)
    a.is_leaf
    b = torch.rand(10, requires_grad=True).cuda()
    b.is_leaf
    c = torch.rand(10, requires_grad=True) + 2
    c.is_leaf
    d = torch.rand(10).cuda()
    d.is_leaf
    e = torch.rand(10).cuda().requires_grad_()
    e.is_leaf
    f = torch.rand(10, requires_grad=True, device="cuda")
    f.is_leaf


def example_torch_tensor_item():
    x = torch.tensor([1.0])
    x.item()


def example_torch_tensor_narrow():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x.narrow(0, 0, 2)
    x.narrow(1, 1, 2)


def example_torch_tensor_new_empty():
    tensor = torch.ones(())
    tensor.new_empty((2, 3))


def example_torch_tensor_new_full():
    tensor = torch.ones((2,), dtype=torch.float64)
    tensor.new_full((3, 4), 3.141592)


def example_torch_tensor_new_ones():
    tensor = torch.tensor((), dtype=torch.int32)
    tensor.new_ones((2, 3))


def example_torch_tensor_new_tensor():
    tensor = torch.ones((2,), dtype=torch.int8)
    data = [[0, 1], [2, 3]]
    tensor.new_tensor(data)


def example_torch_tensor_new_zeros():
    tensor = torch.tensor((), dtype=torch.float64)
    tensor.new_zeros((2, 3))


def example_torch_tensor_permute():
    x = torch.randn(2, 3, 5)
    x.size()
    x.permute(2, 0, 1).size()


def example_torch_tensor_put_():
    src = torch.tensor([[4, 3, 5],
                        [6, 7, 8]])
    src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))


def example_torch_tensor_register_hook():
    v = torch.tensor([0., 0., 0.], requires_grad=True)
    h = v.register_hook(lambda grad: grad * 2)  # double the gradient
    v.backward(torch.tensor([1., 2., 3.]))
    v.grad
    h.remove()  # removes the hook


def example_torch_tensor_repeat():
    x = torch.tensor([1, 2, 3])
    x.repeat(4, 2)
    x.repeat(4, 2, 1).size()


def example_torch_tensor_requires_grad_():
    # Let's say we want to preprocess some saved weights and use
    # the result as new weights.
    saved_weights = [0.1, 0.2, 0.3, 0.25]
    loaded_weights = torch.tensor(saved_weights)
    weights = preprocess(loaded_weights)  # some function
    weights
    # Now, start to record operations done to weights
    weights.requires_grad_()
    out = weights.pow(2).sum()
    out.backward()
    weights.grad


def example_torch_tensor_resize_():
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    x.resize_(2, 2)


def example_torch_tensor_scatter_():
    x = torch.rand(2, 5)
    x
    torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
    z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
    z


def example_torch_tensor_scatter_add_():
    x = torch.rand(2, 5)
    x
    torch.ones(3, 5).scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)


def example_torch_tensor_size():
    torch.empty(3, 4, 5).size()


def example_torch_tensor_sparse_mask():
    nnz = 5
    dims = [5, 5, 2, 2]
    I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                   torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
    V = torch.randn(nnz, dims[2], dims[3])
    size = torch.Size(dims)
    S = torch.sparse_coo_tensor(I, V, size).coalesce()
    D = torch.randn(dims)
    D.sparse_mask(S)


def example_torch_tensor_storage_offset():
    x = torch.tensor([1, 2, 3, 4, 5])
    x.storage_offset()
    x[3:].storage_offset()


def example_torch_tensor_stride():
    x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    x.stride()
    x.stride(-1)


def example_torch_tensor_to():
    tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
    tensor.to(torch.float64)
    cuda0 = torch.device('cuda:0')
    tensor.to(cuda0)
    tensor.to(cuda0, dtype=torch.float64)
    other = torch.randn((), dtype=torch.float64, device=cuda0)
    tensor.to(other, non_blocking=True)


def example_torch_tensor_to_sparse():
    d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
    d
    d.to_sparse()
    d.to_sparse(1)


def example_torch_tensor_tolist():
    a = torch.randn(2, 2)
    a.tolist()
    a[0,0].tolist()


def example_torch_tensor_unfold():
    x = torch.arange(1., 8)
    x
    x.unfold(0, 2, 1)
    x.unfold(0, 2, 2)


def example_torch_tensor_view():
    x = torch.randn(4, 4)
    x.size()
    y = x.view(16)
    y.size()
    z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    z.size()
    a = torch.randn(1, 2, 3, 4)
    a.size()
    b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
    b.size()
    c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
    c.size()