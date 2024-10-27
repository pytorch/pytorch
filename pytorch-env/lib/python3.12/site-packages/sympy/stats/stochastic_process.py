from sympy.core.basic import Basic
from sympy.stats.joint_rv import ProductPSpace
from sympy.stats.rv import ProductDomain, _symbol_converter, Distribution


class StochasticPSpace(ProductPSpace):
    """
    Represents probability space of stochastic processes
    and their random variables. Contains mechanics to do
    computations for queries of stochastic processes.

    Explanation
    ===========

    Initialized by symbol, the specific process and
    distribution(optional) if the random indexed symbols
    of the process follows any specific distribution, like,
    in Bernoulli Process, each random indexed symbol follows
    Bernoulli distribution. For processes with memory, this
    parameter should not be passed.
    """

    def __new__(cls, sym, process, distribution=None):
        sym = _symbol_converter(sym)
        from sympy.stats.stochastic_process_types import StochasticProcess
        if not isinstance(process, StochasticProcess):
            raise TypeError("`process` must be an instance of StochasticProcess.")
        if distribution is None:
            distribution = Distribution()
        return Basic.__new__(cls, sym, process, distribution)

    @property
    def process(self):
        """
        The associated stochastic process.
        """
        return self.args[1]

    @property
    def domain(self):
        return ProductDomain(self.process.index_set,
                             self.process.state_space)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[2]

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Transfers the task of handling queries to the specific stochastic
        process because every process has their own logic of handling such
        queries.
        """
        return self.process.probability(condition, given_condition, evaluate, **kwargs)

    def compute_expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Transfers the task of handling queries to the specific stochastic
        process because every process has their own logic of handling such
        queries.
        """
        return self.process.expectation(expr, condition, evaluate, **kwargs)
