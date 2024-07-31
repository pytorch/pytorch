# not for land

import csv
import os
from typing import List, Any, Optional, Callable
import sys

import torch
import torch.nn.functional as F

from torch.utils._pytree import tree_map

graph_tabular_log = torch._logging.getArtifactLogger(__name__, "graph")

# TODO(future): might be nice to have input shapes here, but needs a refactor since
# they are easiest to get from the subgraph extraction function, but currently summary
# is generated from the subgraph debug function.
summary_headers = ['orig_node_name', 'subgraph_idx', 'lin1_shape', 'lin2_shape', 'subgraph_summary']

def maybe_short_name(torch_fn):
    """
    Tries to format things like

      '<built-in method cat of type object at 0x7f3f7a202c60>'

    as
      
      'torch.cat'
    """
    if hasattr(torch_fn, '__name__'):
        # torch.cat -> cat
        if hasattr(torch, torch_fn.__name__):
            if getattr(torch, torch_fn.__name__) == torch_fn:
                return torch_fn.__name__

        # F.layer_norm -> layer_norm
        if hasattr(F, torch_fn.__name__):
            if getattr(F, torch_fn.__name__) == torch_fn:
                return torch_fn.__name__

        # builtin function mul
        # note: there is definitely a more generic way to do this
        if torch_fn.__name__ == 'mul':
            return 'mul'
        if torch_fn.__name__ == 'add':
            return 'add'
        
        # activation modules 
        # note: there is definitely a more generic way to do this
        if 'torch.nn.modules.activation' in str(torch_fn):
            return torch_fn.__name__

    return torch_fn

def get_meta_val(n: torch.fx.Node):
    # from https://fburl.com/code/hcwdl994
    meta_val = n.meta.get('val', n.meta.get('tensor_meta', n.meta.get('example_value', None)))
    return meta_val

def get_stack_summary(n: torch.fx.Node):
    # from https://fburl.com/code/yify7y7f
    if n.stack_trace:
        parsed_stack_trace = torch.fx.graph._parse_stack_trace(n.stack_trace)
        summary = parsed_stack_trace.get_summary_str()
        return summary
    return None

def is_first_node_of_dual_linear(gm: torch.fx.GraphModule, n: torch.fx.Node):
    first_user = list(n.users.items())[0][0]
    if first_user.op == 'call_module':
        first_user_mod = getattr(gm, first_user.target)
        if type(first_user_mod) == torch.nn.Linear:
            return True
    return False

def debug_single_linear(
    gm: torch.fx.GraphModule, 
    linear_node: torch.fx.Node,
    linear_mod: torch.nn.Module,
    debug_logs_filename: str,
    subgraph_idx: int,
    summary_results: List[Any],
):
    def printme(s):
        # write both to stdout and log file
        graph_tabular_log.debug(s)
        with open(debug_logs_filename, 'a') as f:
            f.write(s + '\n') 

    printme(f'\ndebugging linear {linear_node.target}')
    printme(f'\ndebugging details\n')

    prev_input_shape = None
    prev_node_type = None
    cur_linear_size = linear_mod.in_features, linear_mod.out_features
    cur_linear_2_size = None, None
    next_node_types = []

    # look at the preceding activation
    for prev_n in linear_node.all_input_nodes:
        # to get the shape of the input, we need to look at the previous node
        for prev_prev_n in prev_n.all_input_nodes:
            prev_input_shape = get_meta_val(prev_prev_n).shape
            printme(f'prev input shape: {prev_input_shape}')
        printme(f'prev node: {prev_n.format_node()}')
        printme(f'prev stack_summary: {get_stack_summary(prev_n)}')
        if prev_n.op == 'call_module':
            mod = getattr(gm, prev_n.target)
            prev_node_type = type(mod)
            printme(f'prev mod: {mod}')
        else:
            prev_node_type = prev_n.target

    # print info about current linear
    printme(f'cur_linear node: {linear_node.format_node()}')
    printme(f'cur_linear mod: {linear_mod}')
    printme(f'cur_linear stack_summary: {get_stack_summary(linear_node)}')

    # if there is a dual linear, print that too
    linear_node_to_use = linear_node
    dual_linear = False
    if is_first_node_of_dual_linear(gm, linear_node):
        dual_linear = True
        linear_node_2 = list(linear_node.users.items())[0][0]
        linear_mod_2 = getattr(gm, linear_node_2.target)
        cur_linear_2_size = linear_mod_2.in_features, linear_mod_2.out_features
        printme(f'cur_linear 2 node: {linear_node_2.format_node()}')
        printme(f'cur_linear 2 mod: {linear_mod_2}')
        printme(f'cur_linear 2 stack_summary: {get_stack_summary(linear_node_2)}')
        linear_node_to_use = linear_node_2

    # look at the subsequent ops
    # note: sometimes this is a view, so might need to look farther
    printme(f'num users: {len(linear_node_to_use.users)}')
    for next_n, _ in linear_node_to_use.users.items():
        for next_n_input in next_n.all_input_nodes:
            printme(f'next input shape: {get_meta_val(next_n_input).shape}')
        printme(f'next node: {next_n.format_node()}')
        printme(f'next stack_summary: {get_stack_summary(next_n)}')
        if next_n.op == 'call_module':
            mod = getattr(gm, next_n.target)
            printme(f'next mod: {mod}')
            next_node_types.append(type(mod))
        else:
            next_node_types.append(next_n.target)

    printme('\ndebugging summary\n')
    if not dual_linear:
        linear_shape_str = f'{cur_linear_size}'
        linear_str = 'Linear'
    else:
        linear_shape_str = f'{cur_linear_size} {cur_linear_2_size}'
        linear_str = 'Linear -> Linear'
    printme(f'input_shape {prev_input_shape}, (K, N) {linear_shape_str}')
    subgraph_summary = f'{maybe_short_name(prev_node_type)} -> {linear_str} -> {[maybe_short_name(t) for t in next_node_types]}'
    printme(subgraph_summary)
    printme('\n')

    summary_result = [
        linear_node.target,  # orig_node_name 
        subgraph_idx,
        cur_linear_size,
        cur_linear_2_size,
        subgraph_summary,
    ]
    summary_results.append(summary_result)


    # what we want:
    #
    # summary for quickly understanding opportunity size wrt float8
    # * (a) info for gemm speedup estimation: bsz, input shape, gemm M, K, N
    # * (b) info for needed f8 features: prev_op -> cur_op -> [next_ops]
    #   * for modules, display module type
    #   * for functions, display short function name
    # * info for creating a microbenchmark with representative float8 fusions
    #   is same as (b)
    #
    # long form info for further debugging
    # * input into previous node, with shape and stride
    # * previous node, if module then display type
    # * linear, display weight shape
    #   * display weight shape
    #   * display modeling code location, if available
    # * next node, if module then display type

def extract_linear_subgraph(
    old_gm: torch.fx.GraphModule, 
    old_linear_node: torch.fx.Node,
    old_linear_mod: torch.nn.Module,
    subgraph_save_filename: str,
) -> None:
    """
    Input: a GraphModule with a `linear_node` calling `linear_mod`.

    This function does the following:
    * find the subgraph prev_op -> linear_node -> [*next_ops]
    * create a new GraphModule containing this subgraph
    * save it to disk for further debugging
    """

    # to start, create a new module which just calls the linear
    new_m = torch.nn.Sequential(old_linear_mod)
    new_gm = torch.fx.symbolic_trace(new_m)
    new_g = new_gm.graph
    new_linear_node = list(new_gm.graph.nodes)[1]
    printme = graph_tabular_log.debug
    printme(f'new_gm: {new_gm}')
    printme(f'new_linear_node: {new_linear_node}')

    # copy the linear metadata over
    new_linear_node.meta = old_linear_node.meta
    new_linear_node.args[0].meta = old_linear_node.args[0].meta

    # 
    # step 1: add the preceding activation node
    #
    # before: input -> linear
    # after: input_args -> prev_op -> linear

    # add the node inputs as placeholders, and copy the non-node inputs as is
    prev_old_arg_to_new_arg = {}
    def prev_node_map_arg(old_arg):
        if isinstance(old_arg, torch.fx.Node):
            if old_arg in prev_old_arg_to_new_arg:
                return prev_old_arg_to_new_arg[old_arg]

            with new_g.inserting_before(new_linear_node):
                new_arg = new_g.placeholder(old_arg.name)
                # copy the metadata over
                new_arg.meta = old_arg.meta
            prev_old_arg_to_new_arg[old_arg] = new_arg
            return new_arg
        return old_arg

    assert len(old_linear_node.all_input_nodes) == 1, 'unsupported'
    old_prev_node = old_linear_node.all_input_nodes[0]

    if old_prev_node.op == 'call_module':
        prev_mod = getattr(old_gm, old_prev_node.target)
        new_name = 'prev_mod'
        setattr(new_gm, new_name, prev_mod)
        new_args = tree_map(prev_node_map_arg, old_prev_node.args)
        new_kwargs = tree_map(prev_node_map_arg, old_prev_node.kwargs)
        with new_g.inserting_before(new_linear_node):
            new_prev_node = new_g.call_module(new_name, new_args, new_kwargs)

    elif old_prev_node.op == 'call_function':
        new_args = tree_map(prev_node_map_arg, old_prev_node.args)
        new_kwargs = tree_map(prev_node_map_arg, old_prev_node.kwargs)
        with new_g.inserting_before(new_linear_node):
            new_prev_node = new_g.call_function(old_prev_node.target, new_args, new_kwargs)

    else:
        raise AssertionError('unsupported')

    prev_placeholder = new_linear_node.args[0]
    new_linear_node.args = (new_prev_node,)
    new_g.erase_node(prev_placeholder)

    new_prev_node.meta = old_prev_node.meta
    new_gm.recompile()

    #
    # step 2 (optional): if there is a dual linear, handle it in a single subgraph
    #
    # before: input_args -> prev_op -> linear
    # after:  input_args -> prev_op -> linear -> linear2
    #
    # then, in step 3, next_ops will be after linear2

    # we still need to refer to the first linear in some places,
    # save it
    old_old_linear_node = old_linear_node
    first_new_linear_node = new_linear_node

    if is_first_node_of_dual_linear(old_gm, old_linear_node):
        old_first_user = list(old_linear_node.users.items())[0][0]
        old_first_user_mod = getattr(old_gm, old_first_user.target)
        printme('DUAL LINEAR')
        dual_linear_name = 'dual_linear'
        setattr(new_gm, dual_linear_name, old_first_user_mod)
        new_args, new_kwargs = (new_linear_node,), {}

        with new_g.inserting_after(new_linear_node):
            new_dual_linear_node = new_g.call_module(dual_linear_name, new_args, new_kwargs)
        new_dual_linear_node.meta = old_first_user.meta

        # make the following code treat the second linear as the root
        new_linear_node = new_dual_linear_node
        old_linear_node = old_first_user

    #
    # step 3: add the subsequent nodes (can be multiple users)
    #
    # before: input_args -> prev_op -> linear
    # after: input_args -> prev_op -> linear -> next_op_1
    #                                        -> ...
    #                                        -> next_op_n
    
    # create last_node to ensure graph order matches the original if there
    # are multiple users of linear's output
    new_last_node = new_linear_node
    new_output_nodes = []
    next_old_arg_to_new_arg = {}

    def next_node_map_arg(old_arg):
        if isinstance(old_arg, torch.fx.Node):
            if old_arg in next_old_arg_to_new_arg:
                # handle the same arg being used multiple times
                return next_old_arg_to_new_arg[old_arg]
            if old_arg == old_linear_node:
                next_old_arg_to_new_arg[old_arg] = new_linear_node
                return new_linear_node
            elif old_arg == old_old_linear_node:
                next_old_arg_to_new_arg[old_arg] = first_new_linear_node
                return first_new_linear_node
            elif old_arg == old_prev_node:
                next_old_arg_to_new_arg[old_arg] = new_prev_node
                return new_prev_node
            elif old_arg in prev_old_arg_to_new_arg:
                return prev_old_arg_to_new_arg[old_arg]
            else:
                # this is something else, make it a graph input
                with new_g.inserting_before(new_linear_node):
                    new_arg = new_g.placeholder(old_arg.name)
                    # copy the metadata over
                    new_arg.meta = old_arg.meta
                next_old_arg_to_new_arg[old_arg] = new_arg
                return new_arg
        return old_arg

    next_node_is_output = False
    for counter, (old_next_n, _) in enumerate(old_linear_node.users.items()):
        if old_next_n.op == 'output':
            # nothing to do
            next_node_is_output = True
            break

        new_args = tree_map(next_node_map_arg, old_next_n.args)
        new_kwargs = tree_map(next_node_map_arg, old_next_n.kwargs)
        if old_next_n.op == 'call_function':
            with new_g.inserting_after(new_last_node):
                new_next_n = new_g.call_function(
                    old_next_n.target,
                    new_args,
                    new_kwargs,
                )
            new_output_nodes.append(new_next_n)
            new_last_node = new_next_n
        elif old_next_n.op == 'call_module':
            prev_mod = getattr(old_gm, old_next_n.target)
            new_name = f'next_mod_{counter}'
            setattr(new_gm, new_name, prev_mod)
            with new_g.inserting_after(new_last_node):
                new_next_n = new_g.call_module(new_name, new_args, new_kwargs)
            new_output_nodes.append(new_next_n)
            new_last_node = new_next_n
        else:
            assert False, 'unsupported'
        new_next_n.meta = old_next_n.meta
        new_gm.recompile()
        printme(f'after adding next_node, {new_gm}')

    if not next_node_is_output:
        # reroute graph outputs from `linear` to `new_output_nodes`
        cur_output_node = list(new_g.nodes)[-1]
        printme(f'cur_output_node: {cur_output_node.format_node()}')

        if len(new_output_nodes) == 1:
            new_output_node = new_g.output(new_output_nodes[0])
        else:
            new_output_node = new_g.output(tuple(new_output_nodes))
        printme(f'new_output_node: {cur_output_node.format_node()}')
        new_g.erase_node(cur_output_node)
        new_gm.recompile()
        printme(f'after new output, {new_gm}')

    # ensure every node has metas
    for n in new_g.nodes:
        if n.op == 'output':
            continue
        assert n.meta is not None and n.meta != {}, f'{n}.meta is {n.meta}!'

    test_inputs = []
    for node in new_g.nodes:
        if node.op != 'placeholder':
            continue
        meta = get_meta_val(node)
        test_inputs.append(torch.randn(*meta.shape, dtype=meta.dtype, device='cuda'))
        printme(f'input shape: {test_inputs[-1].shape}')

    # save subgraph and inputs
    torch.save((new_gm, test_inputs), subgraph_save_filename)

    # test fwd+bwd
    test_output = new_gm(*test_inputs)
    # Unified way to test backward, will work as long as all outputs are of the
    # same shape
    test_output = torch.cat([*test_output], dim=0)
    test_output.sum().backward()

    # Note: cannot verify runnable after save/load here, because loading
    # from disk in this file seems to try to use device meta as we are inside
    # of dynamo tracing. Need to load from a separate process to properly test.

def print_and_append_to_logs(logger, filename, s):
    logger.debug(s)
    with open(filename, 'a') as f:
        f.write(s + '\n') 

def debug_linears_for_float8(
    g: torch.fx.Graph,
    target_folder: str,
    linear_mod_filter_fn: Optional[Callable] = None,
) -> None:
    """
    This function:
    1. looks for subgraphs containing `torch.nn.Linear` modules, including the preceding
       and subsequent ops
    2. for each found subgraph
       - extracts metadata about the subgraph (ops, shapes, modeling code location) and saves it to disk
       - extracts it into a new `torch.fx.Graphmodule` instance and saves it to disk, this
         can then be loaded elsewhere to run microbenchmarks

    Inputs:
    - `g` - the graph to debug, assumed to come from dynamo's pre-dispatch trace and have torch IR
    - `target_folder` - the folder to save metadata and microbenchmarks to, note that all folder
      content is overwritten every time the script is run. The contents of this folder will be:
        target_folder/
          debug_logs.txt
          skip_logs.txt
          summary.csv
          subgraph_with_inputs_0.pt
          ...
          subgraph_with_inputs_(n-1).pt
    - `linear_mod_filter_fn`: optional filtering function on linears, if it returns false then subgraph
      extraction is skipped for that linear

    Format of summary.csv (column: example_value):
      orig_node_name: fn_1
      subgraph_idx: 0
      lin1_shape: (2, 3)
      lin2_shape: (3, 4)  # only applies to dual linear subgraphs
      subgraph_summary: ReLU -> Linear -> ["cat"]

    Format of subgraph_with_inputs_0.pt: Tuple[nn.Module, Tuple[torch.tensor]]
    """
    assert not torch._dynamo.config.inline_inbuilt_nn_modules, \
        'torch._dynamo.config.inline_inbuilt_nn_modules must be disabled for this script to work!'

    # ensure target folder exists
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    # ensure target folder only has file extensions we could have written
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if not (file.endswith('.txt') or file.endswith('.pt') or file.endswith('.swp') or file.endswith('.csv') or file.endswith('.json')):
                raise AssertionError(f'unknown file in target_dir: {file}')

    # delete any existing files from previous run for this target_folder
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            os.unlink(os.path.join(root, file))

    debug_logs_filename = os.path.join(target_folder, 'debug_logs.txt')
    skip_logs_filename = os.path.join(target_folder, 'skip_logs.txt')
    summary_filename = os.path.join(target_folder, 'summary.csv')
    summary_results = [summary_headers]

    gm = g.owning_module
    assert gm is not None, 'unsupported, gm needs to be specified'
    printme = graph_tabular_log.debug
    graph_tabular_log.debug("\nstarting linear debug\n")

    def log_skip_linear(n, mod, reason):
        print_and_append_to_logs(graph_tabular_log, skip_logs_filename, f'SKIP: {reason}')
        print_and_append_to_logs(graph_tabular_log, skip_logs_filename, f'node: {n.format_node()}')
        print_and_append_to_logs(graph_tabular_log, skip_logs_filename, f'node.meta: {get_meta_val(n)}')
        print_and_append_to_logs(graph_tabular_log, skip_logs_filename, f'node.stack: {get_stack_summary(n)}')
        print_and_append_to_logs(graph_tabular_log, skip_logs_filename, f'mod: {mod}')
        print_and_append_to_logs(graph_tabular_log, skip_logs_filename, f'\n')

    subgraph_idx = 0
    for n in gm.graph.nodes:
        if n.op != 'call_module':
            continue

        # check for linear
        module_instance = getattr(gm, n.target)
        if type(module_instance) != torch.nn.Linear:
            continue

        if linear_mod_filter_fn is not None and not linear_mod_filter_fn(module_instance):
            log_skip_linear(n, module_instance, 'failed filter function')
            continue

        # Note: we special case for linear -> linear,
        # so if we are at the second linear then skip debug/extract to avoid duplication
        is_second_linear_of_dual_linear = (
            n.args[0].op == 'call_module' and
            type(getattr(gm, n.args[0].target)) == torch.nn.Linear and
            len(n.args[0].users) == 1
        )
        if is_second_linear_of_dual_linear:
            log_skip_linear(n, module_instance, 'second of dual linear')
            continue

        # for now, the case where the linear's input is a graph input is not supported
        is_input_placeholder = n.args[0].op == 'placeholder'
        if is_input_placeholder:
            log_skip_linear(n, module_instance, 'input is placeholder')
            continue

        debug_single_linear(gm, n, module_instance, debug_logs_filename, subgraph_idx, summary_results)
        subgraph_save_filename = os.path.join(target_folder, f'subgraph_with_inputs_{subgraph_idx}.pt')
        extract_linear_subgraph(gm, n, module_instance, subgraph_save_filename)
        subgraph_idx += 1

    with open(summary_filename, 'w') as f:
        csv.writer(f).writerows(summary_results)

    graph_tabular_log.debug("\nending linear debug\n")
