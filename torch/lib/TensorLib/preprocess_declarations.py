

def exclude(d):
    return 'only_register' in d

def run(declarations):
    declarations = [d for d in declarations if not exclude(d)]
    return declarations
