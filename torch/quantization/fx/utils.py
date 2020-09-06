import re

# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]


def graph_pretty_str(g, shorten=True) -> str:
    """Returns a printable representation of the ops in the graph of g.
    If shorten is True, tries to abbreviate fields.
    """
    built_in_func_re = re.compile('<built-in function (.*)>')
    built_in_meth_re = re.compile('<built-in method (.*) of type.*>')
    op_dict = {
        'placeholder': 'plchdr',
        'get_param': 'gt_prm',
        'call_function': 'cl_fun',
        'call_module': 'cl_mod',
        'call_method': 'cl_meth',
    }

    max_lens = {}
    col_names = ("name", "op", "target", "args", "kwargs")
    for s in col_names:
        max_lens[s] = len(s)

    results = []
    for n in g.nodes:

        # activation_post_process_0 -> obs_0
        name = str(n.name)
        if shorten:
            name = name.replace("activation_post_process", "obs")

        op = str(n.op)
        # placeholder -> plchdr, and so on
        if shorten and op in op_dict:
            op = op_dict[op]

        target = str(n.target)
        # <built-in function foo> -> <bi_fun foo>, and so on
        if shorten:
            built_in_func = built_in_func_re.search(target)
            if built_in_func:
                target = f"<bi_fun {built_in_func.group(1)}>"
            built_in_meth = built_in_meth_re.search(target)
            if built_in_meth:
                target = f"<bi_meth {built_in_meth.group(1)}>"
            target = target.replace("activation_post_process", "obs")

        args = str(n.args)
        if shorten:
            args = args.replace("activation_post_process", "obs")

        kwargs = str(n.kwargs)

        # calculate maximum length of each column, so we can tabulate properly
        for k, v in zip(col_names, (name, op, target, args, kwargs)):
            max_lens[k] = max(max_lens[k], len(v))
        results.append([name, op, target, args, kwargs])

    res_str = ""
    format_str = "{:<{name}} {:<{op}} {:<{target}} {:<{args}} {:<{kwargs}}\n"
    res_str += format_str.format(*col_names, **max_lens)
    for result in results:
        res_str += format_str.format(*result, **max_lens)

    # print an exra note on abbreviations which change attribute names,
    # since users will have to un-abbreviate for further debugging
    if shorten:
        res_str += "*obs_{n} = activation_post_process_{n}\n"
    return res_str
