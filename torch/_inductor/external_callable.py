import inspect

external_matmul = []

def register_external_matmul_call(func):
    if not callable(func):
        raise Exception(f'{func} is not a callable')
    params = inspect.signature(func).parameters
    if len(params) != 3:
        raise Exception(f'required 3 params but got {len(params)}')
    external_matmul.append(func)
