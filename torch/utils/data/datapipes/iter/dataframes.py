from torch.utils.data import IterDataPipe, functional_datapipe, DFIterDataPipe
from typing import Iterator, Optional, Sized, Tuple, TypeVar
import pandas
import random

# TODO(VitalyFedyunin): Add error when two different traces get combined


class DataFrameTracedOps(DFIterDataPipe):
    def __init__(self, source_datapipe, output_var):
        self.source_datapipe = source_datapipe
        self.output_var = output_var

    def __iter__(self):
        for item in self.source_datapipe:
            yield self.output_var.calculate_me(item)




@functional_datapipe('dataframes_per_row', is_df = True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe
    
    def __iter__(self):
        for df in self.source_datapipe:
            for i in range(len(df.index)):
                yield df[i:i+1]


@functional_datapipe('dataframes_concat', is_df = True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, batch = 3):
        self.source_datapipe = source_datapipe
        self.batch = batch
    
    def __iter__(self):
        buffer = []
        for df in self.source_datapipe:
            buffer.append(df)
            if len(buffer) == self.batch:
                yield pandas.concat(buffer)
                buffer = []
        if len(buffer):
            yield pandas.concat(buffer)



@functional_datapipe('dataframes_shuffle', is_df = True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe
    
    def __iter__(self):
        size = None
        all_buffer = []
        for df in self.source_datapipe:
            if size is None:
                size = len(df.index)
            for i in range(len(df.index)):
                all_buffer.append(df[i:i+1])
        random.shuffle(all_buffer)
        buffer = []
        for df in all_buffer:
            buffer.append(df)
            if len(buffer) == size:
                yield pandas.concat(buffer)
                buffer = []
        if len(buffer):
            yield pandas.concat(buffer)

        
@functional_datapipe('dataframes_filter', is_df = True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, filter_fn):
        self.source_datapipe = source_datapipe
        self.filter_fn = filter_fn
    
    def __iter__(self):
        size = None
        all_buffer = []
        filter_res = []
        for df in self.source_datapipe:
            if size is None:
                size = len(df.index)
            for i in range(len(df.index)):
                all_buffer.append(df[i:i+1])
                filter_res.append(self.filter_fn(df.iloc[i]))
        
        # random.shuffle(all_buffer)

        # print('filter repack as', size)

        buffer = []
        for df, res in zip(all_buffer, filter_res):
            # print(df)
            # print(res)
            if res:
                buffer.append(df)
                if len(buffer) == size:
                    yield pandas.concat(buffer)
                    buffer = []
        if len(buffer):
            yield pandas.concat(buffer)



@functional_datapipe('to_dataframes_pipe', is_df = True)
class ExampleAggregateAsDataFrames(DFIterDataPipe):
    def __init__(self, source_datapipe, dataframe_size=10, columns=None):
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.dataframe_size = dataframe_size

    def _as_list(self, item):
        try:
            return [i for i in item]
        except:
            return [item]

    def __iter__(self):
        aggregate = []
        for item in self.source_datapipe:
            aggregate.append(self._as_list(item))
            if len(aggregate) == self.dataframe_size:
                yield pandas.DataFrame(aggregate, columns=self.columns)
                aggregate = []
        if len(aggregate) > 0:
            yield pandas.DataFrame(aggregate, columns=self.columns)

DATAPIPES_OPS = ['dataframes_filter','map','to_datapipe','shuffle', 'concat', 'batch', 'dataframes_per_row','dataframes_concat','dataframes_shuffle']

class Capture(object):
    # All operations are shared across entire InitialCapture, need to figure out what if we join two captures
    ctx = None

    def __init__(self):
        self.ctx = {'operations': [], 'variables': []}

    def __str__(self):
        return self.ops_str()

    def ops_str(self):
        res = ""
        for op in self.ctx['operations']:
            if len(res) > 0:
                res += "\n"
            res += str(op)
        return res

    def __getattr__(self, attrname):
        if attrname == 'kwarg':
            raise Exception('no kwargs!')
        if attrname in DATAPIPES_OPS:
            return (self.as_datapipe()).__getattr__(attrname)
        return CaptureGetAttr(self, attrname, ctx=self.ctx)

    # def __setattr__(self, name, value):
    #   pass

    def __getitem__(self, key):
        return CaptureGetItem(self, key, ctx=self.ctx)

    def __setitem__(self, key, value):
        self.ctx['operations'].append(
            CaptureSetItem(self, key, value, ctx=self.ctx))

    # def __call__(self, *args, **kwargs):
    #   pass

    def __add__(self, add_val):
        res = CaptureAdd(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        #
        self.ctx['operations'].append(
            CaptureVariableAssign(variable = var, value = res, ctx=self.ctx))
        return var

    def __sub__(self, add_val):
        res = CaptureSub(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx['operations'].append(
            CaptureVariableAssign(variable = var, value = res, ctx=self.ctx))
        return var

    def __mul__(self, add_val):
        res = CaptureMul(self, add_val, ctx=self.ctx)
        # print('captured mul', res)
        var = CaptureVariable(res, ctx=self.ctx)
        t = CaptureVariableAssign(variable = var, value = res, ctx=self.ctx)
        self.ctx['operations'].append(t)
        return var

    def as_datapipe(self):
        return DataFrameTracedOps(
            self.ctx['variables'][0].source_datapipe, self)

    def raw_iterator(self):
        # print('iterator called')
        return self.as_datapipe().__iter__()

    def __iter__(self):
        # print('iterator called')
        return self.dataframes_per_row().as_datapipe().__iter__()

    def batch(self, batch_size = 10):
        # print('batch called', batch_size)
        dp = self.dataframes_per_row().dataframes_concat(batch_size)
        dp = dp.as_datapipe().batch(1)
        dp._dp_contains_dataframe = True
        return dp

    def shuffle(self, *args, **kwargs):
        return self.dataframes_shuffle(*args, **kwargs)

    def filter(self, *args, **kwargs):
        return self.dataframes_filter(*args, **kwargs)

class CaptureF(Capture):
    # kwargs = {}

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
        # print('executing',self)
        # print('calculating value of', self.kwarg['value'])
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

    def calculate_me(self, dataframe):
        self.ctx['variables'][0].calculated_value = dataframe
        for op in self.ctx['operations']:
            # print('executing ', str(op))
            op.execute()
        return self.calculated_value







class CaptureInitial(CaptureVariable):

    def __init__(self):
        new_ctx = {'operations': [], 'variables': []}
        super().__init__(None, new_ctx)
        self.name = 'input_%s' % self.name


class CaptureGetItem(Capture):
    left = None
    key = None

    def __init__(self, left, key, ctx):
        self.ctx = ctx
        self.left = left
        self.key = key

    def __str__(self):
        return "%s[%s]" % (self.left, get_val(self.key))

    def execute(self):
        # print('getting value of', str(self.left))
        return (self.left.execute())[self.key]


class CaptureSetItem(Capture):
    left = None
    key = None
    value = None

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
    name = None

    def __init__(self, src, name, ctx):
        
        self.ctx = ctx
        self.src = src
        self.name = name
        # print('getattr captured',self.name)

    def __str__(self):
        return "%s.%s" % (self.src, self.name)

    def execute(self):
        # print('getting attr', self)
        val = get_val(self.src)
        return getattr(val, self.name)


def get_val(capture):
    if isinstance(capture, Capture):
        return capture.execute()
    elif isinstance(capture, str):
        return '"%s"' % capture
    else:
        return capture


@functional_datapipe('trace_as_dataframe')
class DataFrameTracer(CaptureInitial, IterDataPipe):
    source_datapipe = None

    def __init__(self, source_datapipe):
        super().__init__()
        self.source_datapipe = source_datapipe
