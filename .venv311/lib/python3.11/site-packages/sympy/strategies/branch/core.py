""" Generic SymPy-Independent Strategies """


def identity(x):
    yield x


def exhaust(brule):
    """ Apply a branching rule repeatedly until it has no effect """
    def exhaust_brl(expr):
        seen = {expr}
        for nexpr in brule(expr):
            if nexpr not in seen:
                seen.add(nexpr)
                yield from exhaust_brl(nexpr)
        if seen == {expr}:
            yield expr
    return exhaust_brl


def onaction(brule, fn):
    def onaction_brl(expr):
        for result in brule(expr):
            if result != expr:
                fn(brule, expr, result)
            yield result
    return onaction_brl


def debug(brule, file=None):
    """ Print the input and output expressions at each rule application """
    if not file:
        from sys import stdout
        file = stdout

    def write(brl, expr, result):
        file.write("Rule: %s\n" % brl.__name__)
        file.write("In: %s\nOut: %s\n\n" % (expr, result))

    return onaction(brule, write)


def multiplex(*brules):
    """ Multiplex many branching rules into one """
    def multiplex_brl(expr):
        seen = set()
        for brl in brules:
            for nexpr in brl(expr):
                if nexpr not in seen:
                    seen.add(nexpr)
                    yield nexpr
    return multiplex_brl


def condition(cond, brule):
    """ Only apply branching rule if condition is true """
    def conditioned_brl(expr):
        if cond(expr):
            yield from brule(expr)
        else:
            pass
    return conditioned_brl


def sfilter(pred, brule):
    """ Yield only those results which satisfy the predicate """
    def filtered_brl(expr):
        yield from filter(pred, brule(expr))
    return filtered_brl


def notempty(brule):
    def notempty_brl(expr):
        yielded = False
        for nexpr in brule(expr):
            yielded = True
            yield nexpr
        if not yielded:
            yield expr
    return notempty_brl


def do_one(*brules):
    """ Execute one of the branching rules """
    def do_one_brl(expr):
        yielded = False
        for brl in brules:
            for nexpr in brl(expr):
                yielded = True
                yield nexpr
            if yielded:
                return
    return do_one_brl


def chain(*brules):
    """
    Compose a sequence of brules so that they apply to the expr sequentially
    """
    def chain_brl(expr):
        if not brules:
            yield expr
            return

        head, tail = brules[0], brules[1:]
        for nexpr in head(expr):
            yield from chain(*tail)(nexpr)

    return chain_brl


def yieldify(rl):
    """ Turn a rule into a branching rule """
    def brl(expr):
        yield rl(expr)
    return brl
