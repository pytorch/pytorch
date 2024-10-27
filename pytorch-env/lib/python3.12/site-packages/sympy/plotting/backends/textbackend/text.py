import sympy.plotting.backends.base_backend as base_backend
from sympy.plotting.series import LineOver1DRangeSeries
from sympy.plotting.textplot import textplot


class TextBackend(base_backend.Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def show(self):
        if not base_backend._show:
            return
        if len(self._series) != 1:
            raise ValueError(
                'The TextBackend supports only one graph per Plot.')
        elif not isinstance(self._series[0], LineOver1DRangeSeries):
            raise ValueError(
                'The TextBackend supports only expressions over a 1D range')
        else:
            ser = self._series[0]
            textplot(ser.expr, ser.start, ser.end)

    def close(self):
        pass
