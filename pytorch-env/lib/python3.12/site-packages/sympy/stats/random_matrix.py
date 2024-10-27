from sympy.core.basic import Basic
from sympy.stats.rv import PSpace, _symbol_converter, RandomMatrixSymbol

class RandomMatrixPSpace(PSpace):
    """
    Represents probability space for
    random matrices. It contains the mechanics
    for handling the API calls for random matrices.
    """
    def __new__(cls, sym, model=None):
        sym = _symbol_converter(sym)
        if model:
            return Basic.__new__(cls, sym, model)
        else:
            return Basic.__new__(cls, sym)

    @property
    def model(self):
        try:
            return self.args[1]
        except IndexError:
            return None

    def compute_density(self, expr, *args):
        rms = expr.atoms(RandomMatrixSymbol)
        if len(rms) > 2 or (not isinstance(expr, RandomMatrixSymbol)):
            raise NotImplementedError("Currently, no algorithm has been "
                    "implemented to handle general expressions containing "
                    "multiple random matrices.")
        return self.model.density(expr)
