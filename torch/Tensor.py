class RealTensor(C.RealTensorBase):
    def __str__(self):
        return "RealTensor"

    def __repr__(self):
        return str(self)

    def type(self, t):
        current = "torch." + self.__class__.__name__
        if not t:
            return current
        if t == current:
            return self
        _, _, typename = t.partition('.')
        assert hasattr(torch, typename)
        return getattr(torch, typename)(self.size()).copy(self)

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
