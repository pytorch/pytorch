class CalculusMethods(object):
    pass

def defun(f):
    setattr(CalculusMethods, f.__name__, f)
    return f
