import torch
from functorch import jacfwd
class FunctionWrapper(object):
    def __init__(self, fun):  # note that function can be a lambda expression
        self._fun = fun
        self.fevals = 0
        self.cur_x = None
        self.cur_y = None
        self.input_segment = None
        self.constant_dict = None

    def __call__(self, v, **kwargs):
        self.fevals += 1
        # self.cur_x = v.view(-1).requires_grad_()
        self.cur_x = v.view(-1)
        if self.input_segment is not None:
            x = []
            for i, _ in enumerate(self.input_segment):
                if i > 0:
                    x.append(v[self.input_segment[i - 1]:self.input_segment[i]])
            self.cur_y = self._fun(v, *x, **kwargs)
        else:
            self.cur_y = self._fun(v, self.cur_x, **kwargs)
        return self.cur_y

    def input_constructor(self, *args):  # if has time, convert to kwargs input
        l = []
        self.input_segment = [0]
        cur = 0
        for v in args:
            nv = v.view(-1)
            l.append(nv)
            cur += nv.size()[0]
            self.input_segment.append(cur)
        x = torch.concat(l).detach().requires_grad_()
        return x

if __name__ == '__main__':
    xx_ = torch.tensor([4.,5.,6.])
    yy_ = torch.tensor([7.,8.])
    def func(xx_, *args):
        #x_ = torch.tensor([1.,2.,3.,4.])
        xx_[:2] = args[0]
        y = args[1]
        return torch.vstack([(xx_**2).sum(),(y**3).sum()])
    funcc = FunctionWrapper(func)
    xx = funcc.input_constructor(xx_[:2],yy_)
    print(torch.autograd.functional.jacobian(funcc,xx))
    print(jacfwd(funcc)(xx))
