from sympy.core.basic import Basic

new = Basic.__new__


def assoc(d, k, v):
    d = d.copy()
    d[k] = v
    return d


basic_fns = {'op': type,
             'new': Basic.__new__,
             'leaf': lambda x: not isinstance(x, Basic) or x.is_Atom,
             'children': lambda x: x.args}

expr_fns = assoc(basic_fns, 'new', lambda op, *args: op(*args))
