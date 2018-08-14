import torch

class SobolEngine(object):
    """
    The ``SobolEngine`` is an engine for generating (scrambled) Sobol sequences.
    Sobol sequences are an example of low discrepancy quasi-random sequences.

    This implementation of an engine for Sobol sequences is capable of sampling sequences up to
    a maximum dimension of 1111. It uses direction numbers to generate these sequences, and these
    numbers have been adapted from `here <http://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-old.1111>`_.

    References:
      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points. Journal of
        Complexity, 14(4):466-489, December 1998.

      - I. M. Sobol. The distribution of points in a cube and the accurate
        evaluation of integrals. Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

    Args:
        dimension (Int): The dimensionality of the sequence to be drawn
        scramble (bool, optional): Setting this to ``True`` will produce scrambled Sobol sequences.
                                   Scrambling is capable of producing better Sobol sequences.
                                   Default: ``False``.
        seed (Int, optional): This is the seed for the scrambling. The seed of the random number
                              generator is set to this, if specified. Default: ``None``

    Examples::

        >>> soboleng = torch.quasirandom.SobolEngine(dimension=5, scramble=False)
        >>> soboleng.draw(3)
        tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.7500, 0.2500, 0.7500],
                [0.2500, 0.7500, 0.2500, 0.7500, 0.2500]])
    """
    MAXBIT = 30

    def __init__(self, dimension, scramble=False, seed=None):
        if dimension > 1111 or dimension < 1:
            raise ValueError("Supported range of dimensionality for SobolEngine is [1, 1111]")

        self.seed = seed
        self.scramble = scramble
        self.dimension = dimension

        self.sobolstate = torch._sobol_engine_initialize_state(torch.zeros(dimension, self.MAXBIT, dtype=torch.long),
                                                               self.dimension)

        if self.scramble:
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed)

            self.shift = torch.mv(torch.randint(2, (self.dimension, self.MAXBIT), generator=g),
                                  torch.pow(2, torch.arange(0, self.MAXBIT, dtype=torch.double))).to(torch.long)

            # TODO: can be replaced with torch.tril(torch.randint(2, (dimension, MAXBIT, MAXBIT)))
            #       once a batched version is introduced
            ltm = torch.randint(2, (self.dimension, self.MAXBIT, self.MAXBIT), dtype=torch.long, generator=g)
            ltm = list(map(lambda x: x.tril(), ltm.unbind(0)))

            self.sobolstate = torch._sobol_engine_scramble(self.sobolstate, ltm, self.dimension)
        else:
            self.shift = torch.zeros(self.dimension, dtype=torch.long)

        self.quasi = self.shift.clone()
        self.num_generated = 0

    def draw(self, n=1):
        """
        Function to draw a sequence of :attr:`n` points from a Sobol sequence. Note that the samples are dependent
        on the previous samples.

        Args:
            n (Int, optional): The length of sequence of points to draw. Default: 1.
        """
        result, self.quasi = torch._sobol_engine_draw(self.quasi, n, self.sobolstate,
                                                      self.dimension, self.num_generated)
        self.num_generated += n
        return result

    def reset(self):
        """
        Function to reset the ``SobolEngine`` to base state.
        """
        self.quasi.copy_(self.shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        """
        Function to fast-forward the state of the ``SobolEngine`` by :attr:`n` steps. This is equivalent to drawing
        :attr:`n` samples without using the samples.

        Args:
            n (Int): The number of steps to fast-forward by.
        """
        self.quasi = torch._sobol_engine_ff(self.quasi, n, self.sobolstate, self.dimension, self.num_generated)
        self.num_generated += n
        return self

    def __repr__(self):
        fmt_string = ['dimension={}'.format(self.dimension)]
        if self.scramble:
            fmt_string += ['scramble=True']
        if self.seed is not None:
            fmt_string += ['seed={}'.format(self.seed)]
        return self.__class__.__name__ + '(' + ', '.join(fmt_string) + ')'
