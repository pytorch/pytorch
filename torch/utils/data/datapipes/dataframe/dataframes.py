from typing import Any, Dict, List

from torch.utils.data import (
    DFIterDataPipe,
    IterDataPipe,
    functional_datapipe,
)

from torch.utils.data.datapipes.dataframe.structures import DataChunkDF

# TODO(VitalyFedyunin): Add error when two different traces get combined


class DataFrameTracedOps(DFIterDataPipe):
    def __init__(self, source_datapipe, output_var):
        self.source_datapipe = source_datapipe
        self.output_var = output_var

    def __iter__(self):
        for item in self.source_datapipe:
            yield self.output_var.apply_ops(item)


#  TODO(VitalyFedyunin): Extract this list from the DFIterDataPipe registred functions
DATAPIPES_OPS = ['_dataframes_as_tuples', 'groupby', '_dataframes_filter', 'map', 'to_datapipe',
                 'shuffle', 'concat', 'batch', '_dataframes_per_row', '_dataframes_concat', '_dataframes_shuffle']


class Capture(object):
    # TODO: All operations are shared across entire InitialCapture, need to figure out what if we join two captures
    ctx: Dict[str, List[Any]]

    def __init__(self):
        self.ctx = {'operations': [], 'variables': []}

    def __str__(self):
        return self._ops_str()

    def _ops_str(self):
        res = ""
        for op in self.ctx['operations']:
            if len(res) > 0:
                res += "\n"
            res += str(op)
        return res

    def __getattr__(self, attrname):
        if attrname == 'kwarg':
            raise Exception('no kwargs!')
        return CaptureGetAttr(self, attrname, ctx=self.ctx)

    def __getitem__(self, key):
        return CaptureGetItem(self, key, ctx=self.ctx)

    def __setitem__(self, key, value):
        self.ctx['operations'].append(
            CaptureSetItem(self, key, value, ctx=self.ctx))

    def __add__(self, add_val):
        res = CaptureAdd(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx['operations'].append(
            CaptureVariableAssign(variable=var, value=res, ctx=self.ctx))
        return var

    def __sub__(self, add_val):
        res = CaptureSub(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx['operations'].append(
            CaptureVariableAssign(variable=var, value=res, ctx=self.ctx))
        return var

    def __mul__(self, add_val):
        res = CaptureMul(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        t = CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        self.ctx['operations'].append(t)
        return var


class CaptureF(Capture):
    def __init__(self, ctx=None, **kwargs):
        if ctx is None:
            self.ctx = {'operations': [], 'variables': []}
        self.ctx = ctx
        self.kwargs = kwargs


class CaptureCall(CaptureF):
    def __str__(self):
        return "{variable}({args},{kwargs})".format(**self.kwargs)

    def execute(self):
        return (get_val(self.kwargs['variable']))(*self.kwargs['args'], **self.kwargs['kwargs'])


class CaptureVariableAssign(CaptureF):
    def __str__(self):
        return "{variable} = {value}".format(**self.kwargs)

    def execute(self):
        self.kwargs['variable'].calculated_value = self.kwargs['value'].execute()


class CaptureVariable(Capture):
    value = None
    name = None
    calculated_value = None
    names_idx = 0

    def __init__(self, value, ctx):
        self.ctx = ctx
        self.value = value
        self.name = 'var_%s' % CaptureVariable.names_idx
        CaptureVariable.names_idx += 1
        self.ctx['variables'].append(self)

    def __str__(self):
        return self.name

    def execute(self):
        return self.calculated_value

    def apply_ops(self, dataframe):
        # TODO(VitalyFedyunin): Make this calculation thread safe (as currently it updates pointer)
        self.ctx['variables'][0].calculated_value = dataframe
        for op in self.ctx['operations']:
            op.execute()
        return self.calculated_value


class CaptureGetItem(Capture):
    left: Capture
    key: Any

    def __init__(self, left, key, ctx):
        self.ctx = ctx
        self.left = left
        self.key = key

    def __str__(self):
        return "%s[%s]" % (self.left, get_val(self.key))

    def execute(self):
        return (self.left.execute())[self.key]


class CaptureSetItem(Capture):
    left: Capture
    key: Any
    value: Capture

    def __init__(self, left, key, value, ctx):
        self.ctx = ctx
        self.left = left
        self.key = key
        self.value = value

    def __str__(self):
        return "%s[%s] = %s" % (self.left, get_val(self.key), self.value)

    def execute(self):
        (self.left.execute())[
            self.key] = self.value.execute()


class CaptureAdd(Capture):
    left = None
    right = None

    def __init__(self, left, right, ctx):
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self):
        return "%s + %s" % (self.left, self.right)

    def execute(self):
        return get_val(self.left) + get_val(self.right)


class CaptureMul(Capture):
    left = None
    right = None

    def __init__(self, left, right, ctx):
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self):
        return "%s * %s" % (self.left, self.right)

    def execute(self):
        return get_val(self.left) * get_val(self.right)


class CaptureSub(Capture):
    left = None
    right = None

    def __init__(self, left, right, ctx):
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self):
        return "%s - %s" % (self.left, self.right)

    def execute(self):
        return get_val(self.left) - get_val(self.right)


class CaptureGetAttr(Capture):
    source = None
    name: str

    def __init__(self, src, name, ctx):
        self.ctx = ctx
        self.src = src
        self.name = name

    def __str__(self):
        return "%s.%s" % (self.src, self.name)

    def execute(self):
        val = get_val(self.src)
        return getattr(val, self.name)


def get_val(capture):
    if isinstance(capture, Capture):
        return capture.execute()
    elif isinstance(capture, str):
        return '"%s"' % capture
    else:
        return capture


class CaptureInitial(CaptureVariable):

    def __init__(self):
        new_ctx: Dict[str, List[Any]] = {'operations': [], 'variables': []}
        super().__init__(None, new_ctx)
        self.name = 'input_%s' % self.name


class CaptureDataFrame(CaptureInitial):
    pass


class CaptureDataFrameWithDataPipeOps(CaptureDataFrame):
    def as_datapipe(self):
        return DataFrameTracedOps(
            self.ctx['variables'][0].source_datapipe, self)

    def raw_iterator(self):
        return self.as_datapipe().__iter__()

    def __iter__(self):
        return iter(self._dataframes_as_tuples())

    def batch(self, batch_size=10, drop_last: bool = False, wrapper_class=DataChunkDF):
        dp = self._dataframes_per_row()._dataframes_concat(batch_size)
        dp = dp.as_datapipe().batch(1, drop_last=drop_last, wrapper_class=wrapper_class)
        dp._dp_contains_dataframe = True
        return dp

    def groupby(self,
                group_key_fn,
                *,
                buffer_size=10000,
                group_size=None,
                guaranteed_group_size=None,
                drop_remaining=False):
        dp = self._dataframes_per_row()
        dp = dp.as_datapipe().groupby(group_key_fn, buffer_size=buffer_size, group_size=group_size,
                                      guaranteed_group_size=guaranteed_group_size, drop_remaining=drop_remaining)
        return dp

    def shuffle(self, *args, **kwargs):
        return self._dataframes_shuffle(*args, **kwargs)

    def filter(self, *args, **kwargs):
        return self._dataframes_filter(*args, **kwargs)

    def __getattr__(self, attrname):  # ?
        if attrname in DATAPIPES_OPS:
            return (self.as_datapipe()).__getattr__(attrname)
        return super().__getattr__(attrname=attrname)


@functional_datapipe('trace_as_dataframe')
class DataFrameTracer(CaptureDataFrameWithDataPipeOps, IterDataPipe):
    source_datapipe = None

    def __init__(self, source_datapipe):
        super().__init__()
        self.source_datapipe = source_datapipe
