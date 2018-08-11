import torch

class SobolEngine(object):

    MAXBIT = 30

    def __init__(self, dimension, scramble=False, seed=None):
        if dimen > 1111 or dimen < 1:
            raise ValueError("Supported range of dimensionality for SobolEngine is [1, 1111]")

        self.seed = seed
        self.scramble = scramble
        self.dimension = dimension

        self.sobolstate = torch.zeros(dimension, self.MAXBIT, dtype=torch.int)
        torch._sobol_engine_initialize_state(self.sobolstate, self.dimension)

        if scramble:
            self.shift = torch.mv(torch.randint(2, (self.dimension, self.MAXBIT)),
                                  torch.pow(2, torch.arange(0, self.MAXBIT))).to(torch.int)

            # TODO: can be replaced with torch.tril(torch.randint(2, (dimension, MAXBIT, MAXBIT)))
            #       once a batched version is introduced
            ltm = [torch.tril(torch.randint(2, (self.MAXBIT, self.MAXBIT))).to(torch.int)
                   for _ in range(0, self.dimension)]

            torch._sobol_engine_scramble(self.sobolstate, ltm, self.dimension)
        else:
            self.shift = torch.zeros(self.dimension, dtype=torch.int)

        self.quasi = self.shift.clone()
        self.num_generated = 0

    def draw(self, n=1):
        result = torch._sobol_engine_draw(n, self.sobolstate, self.quasi, self.dimension, self.num_generated)
        self.num_generated = result[1]
        return result[0]

    def reset(self):
        self.quasi.copy_(self.shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        result = torch._sobol_engine_ff(n, self.sobolstate, self.quasi, self.dimension, self.num_generated)
        self.num_generated = result
        return self
