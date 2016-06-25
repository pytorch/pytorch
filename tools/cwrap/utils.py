def argfilter():
    """Returns a function, that allows to filter out already known arguments.

       self is used only in stateful mode and is always provided.
       _res_new is allocated automatically before call, so it is known.
       CONSTANT arguments are literals.
       Repeated arguments do not need to be specified twice.
    """
    provided = set()
    def is_already_provided(arg):
        ret = False
        ret |= arg.name == 'self'
        ret |= arg.name == '_res_new'
        ret |= arg.type == 'CONSTANT'
        ret |= arg.type == 'EXPRESSION'
        ret |= arg.name in provided
        provided.add(arg.name)
        return ret
    return is_already_provided
