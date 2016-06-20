
class RealTensor(RealTensorBase):
    def new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def type(self, t):
        current = "torch." + self.__class__.__name__
        if not t:
            return current
        if t == current:
            return self
        _, _, typename = t.partition('.')
        # TODO: this is ugly
        assert hasattr(sys.modules['torch'], typename)
        return getattr(sys.modules['torch'], typename)(self.size()).copy(self)

    def typeAs(self, t):
        return self.type(t.type())

    def double(self):
        return self.type('torch.DoubleTensor')

    def float(self):
        return self.type('torch.FloatTensor')

    def long(self):
        return self.type('torch.LongTensor')

    def int(self):
        return self.type('torch.IntTensor')

    def short(self):
        return self.type('torch.ShortTensor')

    def char(self):
        return self.type('torch.CharTensor')

    def byte(self):
        return self.type('torch.ByteTensor')

    def __repr__(self):
        return str(self)

    def __str__(self):
        return _printing.printTensor(self)

    def __iter__(self):
        return iter(map(lambda i: self.select(0, i), pyrange(self.size(0))))

    def split(self, split_size, dim=0):
        result = []
        dim_size = self.size(dim)
        num_splits = math.ceil(dim_size / split_size)
        last_split_size = split_size * num_splits - dim_size or split_size
        def get_split_size(i):
            return split_size if i < num_splits-1 else last_split_size
        return [self.narrow(dim, i*split_size, get_split_size(i)) for i
                in pyrange(0, num_splits)]

    def chunk(self, n_chunks, dim=0):
        split_size = math.ceil(tensor.size(dim)/n_chunks)
        return torch.split(tensor, split_size, dim)

    def tolist(self):
        dim = self.dim()
        if dim == 1:
            return [v for v in self]
        elif dim > 0:
            return [subt.tolist() for subt in self]
        return []

    def view(self, src, *args):
        assert isTensor(src)
        if len(args) == 1 and isStorage(args[0]):
            sizes = args[0]
        else:
            sizes = LongStorage(args)
        sizes = infer_sizes(sizes, src.nElement())

        assert src.isContiguous(), "expecting a contiguous tensor"
        self.set(src.storage(), src.storageOffset(), sizes)
        return self

    def viewAs(self, src, template):
        if not isTensor(src) and isLongStorage(template):
            raise ValueError('viewAs is expecting a Tensor and LongStorage')
        return self.view(src, template.size())

    def permute(self, *args):
        perm = list(args)
        tensor = self
        n_dims = tensor.dim()
        assert len(perm) == n_dims, 'Invalid permutation'
        for i, p in enumerate(perm):
            if p != i and p != -1:
                j = i
                while True:
                    assert 0 <= perm[j] and perm[j] < n_dims, 'Invalid permutation'
                    tensor = tensor.transpose(j, perm[j])
                    perm[j], j = -1, perm[j]
                    if perm[j] == i:
                        break
                perm[j] = -1
        return tensor

    def expandAs(self, src, template=None):
        if template is not None:
            return self.expand(src, template.size())
        return self.expand(src.size())

    def expand(self, src, *args):
        if not isTensor(src):
            if isStorage(src) and len(args) == 0:
                sizes = src
            else:
                # TODO: concat iters
                sizes = LongStorage([src] + list(args))
            src = self
            result = self.new()
        else:
            sizes = LongStorage(args)
            result = self

        src_dim = src.dim()
        src_stride = src.stride()
        src_size = src.size()

        if sizes.size() != src_dim:
            raise ValueError('the number of dimensions provided must equal tensor.dim()')

        # create a new geometry for tensor:
        for i, size in enumerate(src_size):
            if size == 1:
                src_size[i] = sizes[i]
                src_stride[i] = 0
            elif size != sizes[i]:
                raise ValueError('incorrect size: only supporting singleton expansion (size=1)')

        result.set(src.storage(), src.storageOffset(),
                                src_size, src_stride)
        return result

    # TODO: maybe drop this in favour of csub? :(
    def sub(self, *sizes):
        if len(sizes) == 0:
            raise ValueError('sub requires at least two arguments')
        if len(sizes) % 2 != 0:
            raise ValueError('sub requires an even number of arguments')
        result = self
        pairs = int(len(sizes)/2)
        for dim, start, end in zip(pyrange(pairs), sizes[::2], sizes[1::2]):
            dim_size = result.size(dim)
            start = start + dim_size if start < 0 else start
            end = end + dim_size if end < 0 else end
            result = result.narrow(dim, start, end-start+1)
        return result

    def repeatTensor(self, src, *args):
        if not isTensor(src):
            if isStorage(src) and len(args) == 0:
                repeats = src.tolist()
            else:
                repeats = [src] + list(args)
            src = self
            result = self.new()
        else:
            repeats = list(args)
            result = self

        if not src.isContiguous():
            src = src.clone()

        if len(repeats) < src.dim():
            raise ValueError('Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor')

        xtensor = src.new().set(src)
        xsize = xtensor.size().tolist()
        for i in pyrange(len(repeats)-src.dim()):
            xsize = [1] + xsize

        size = LongStorage([a * b for a, b in zip(xsize, repeats)])
        xtensor.resize(LongStorage(xsize))
        result.resize(size)
        urtensor = result.new(result)
        for i in pyrange(xtensor.dim()):
            urtensor = urtensor.unfold(i,xtensor.size(i),xtensor.size(i))
        for i in pyrange(urtensor.dim()-xtensor.dim()):
            xsize = [1] + xsize
        xtensor.resize(LongStorage(xsize))
        xxtensor = xtensor.expandAs(urtensor)
        urtensor.copy(xxtensor)
        return result

    def __add__(self, other):
        return self.clone().add(other)
    __radd__ = __add__

    def __sub__(self, other):
        return self.clone().csub(other)
    __rsub__ = __sub__

    def __mul__(self, other):
        # TODO: isTensor checks many cases, while it might be faster to only
        # see if other is a number. It's a weird thing in Python, so share
        # some THPUtils functions in C namespace in the future.
        if isTensor(other):
            dim_self = self.dim()
            dim_other = other.dim()
            if dim_self == 1 and dim_other == 1:
                return self.dot(other)
            elif dim_self == 2 and dim_other == 1:
                return self.new().mv(self, other)
            elif dim_self == 2 and dim_other == 2:
                return self.new().mm(self, other)
        else:
            return self.clone().mul(other)

    def __rmul__(self, other):
        # No need to check for tensor on lhs - it would execute it's __mul__
        return self.clone().mul(other)

    def __div__(self, other):
        return self.clone().div(other)
    __rdiv__ = __div__

    def __mod__(self, other):
        return self.clone().remainder(other)

    def __neg__(self):
        return self.clone().mul(-1)


