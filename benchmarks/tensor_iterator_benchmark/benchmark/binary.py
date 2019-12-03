import torch
import gc
from . import layout_shape, timing
import itertools

selected_dtypes = [torch.bool, torch.float32, torch.int64]
all_dtypes = torch.testing.get_all_dtypes()
floating_point_dtypes = [x for x in all_dtypes if x.is_floating_point]
integeral_dtypes = set(all_dtypes) - set(floating_point_dtypes)

arithmetic_nodiv = ['add']  # + ['sub', 'mul']
compare = ['lt']  # + ['le', 'gt', 'ge', 'eq', 'ne']
bitwise = ['bitwise_xor']
logical = ['logical_xor']
other = ['atan2']

selected_ops = ['add', 'add_', 'lt', 'lt_']

def filter_illegal(it):
    result = []
    for x in it:
        op, dtype1, dtype2 = x
        a = torch.empty(5, dtype=dtype1, device='cuda')
        b = torch.empty(5, dtype=dtype2, device='cuda')
        try:
            getattr(a, op)(b)
            result.append(x)
        except RuntimeError:
            pass
        except AttributeError:
            pass
    return result

selected_combinations = filter_illegal(itertools.product(selected_ops, selected_dtypes, selected_dtypes))
more_combinations = filter_illegal(itertools.chain(
    itertools.product(arithmetic_nodiv, all_dtypes, all_dtypes),
    itertools.product((x + '_' for x in arithmetic_nodiv), all_dtypes, all_dtypes),
    itertools.product(['div', 'div_'], floating_point_dtypes, floating_point_dtypes),
    itertools.product(compare, all_dtypes, all_dtypes),
    itertools.product((x + '_' for x in compare), all_dtypes, all_dtypes),
    itertools.product(bitwise, integeral_dtypes, integeral_dtypes),
    itertools.product((x + '_' for x in bitwise), integeral_dtypes, integeral_dtypes),
    itertools.product(logical, all_dtypes, all_dtypes),
    itertools.product((x + '_' for x in logical), all_dtypes, all_dtypes),
    itertools.product(other, floating_point_dtypes, floating_point_dtypes),
    itertools.product((x + '_' for x in other), floating_point_dtypes, floating_point_dtypes),
))

def larger_dtype(dtype1, dtype2):
    len1 = layout_shape.sizeof(dtype1)
    len2 = layout_shape.sizeof(dtype2)
    return dtype1 if len1 > len2 else dtype2

def run(more):
    title = "binary op"
    combinations = more_combinations if more else selected_combinations
    for op, dtype1, dtype2 in combinations:

        def setup(device, non_contiguous_size='-inf'):
            return {
                'op': op,
                'dtype1': str(dtype1),
                'dtype2': str(dtype2),
                'layout': name,
                'device': device,
                'non_contiguous_size': non_contiguous_size,
            }

        def benchmark_cpu(factories):
            print('Benchmarking', op, 'with dtype', (dtype1, dtype2), 'and layout', name, 'on cpu')
            data = []
            for factory in factories:
                tensor1 = factory.new(dtype1, 'cpu')
                tensor2 = factory.new(dtype2, 'cpu')
                operator = getattr(tensor1, op)
                one_loop_timer = timing.time_one_loop(lambda: operator(tensor2))
                result = timing.time_func(one_loop_timer, tensor1.numel())
                data.append(({'problem_size': factory.problem_size, 'result': result}))
                del tensor1, tensor2, one_loop_timer, operator
                gc.collect()
            return data

        def benchmark_cuda(factories):
            print('Benchmarking', op, 'with dtype', (dtype1, dtype2), 'and layout', name, 'on cuda')
            data = []
            for factory in factories:
                tensor1 = factory.new(dtype1, 'cuda')
                tensor2 = factory.new(dtype2, 'cuda')
                operator = getattr(tensor1, op)
                one_loop_timer = timing.time_one_loop_cuda(lambda: operator(tensor2))
                result = timing.time_func(one_loop_timer, tensor1.numel())
                data.append(({'problem_size': factory.problem_size, 'result': result}))
                del tensor1, tensor2, one_loop_timer, operator
                gc.collect()
            return data

        f1d, f2d = layout_shape.get(larger_dtype(dtype1, dtype2), more)
        for name, factories in f1d.items():
            if dtype1 is not torch.float16 and dtype2 is not torch.float16:
                yield (title, {'setup': setup('cpu'), 'data': benchmark_cpu(factories)})
            yield (title, {'setup': setup('cuda'), 'data': benchmark_cuda(factories)})

        for name, d in f2d.items():
            for non_contiguous_size, factories in d.items():
                if dtype1 is not torch.float16 and dtype2 is not torch.float16:
                    yield (title, {'setup': setup('cpu', non_contiguous_size), 'data': benchmark_cpu(factories)})
                yield (title, {'setup': setup('cuda', non_contiguous_size), 'data': benchmark_cuda(factories)})
