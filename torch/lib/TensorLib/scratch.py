import re


def to_environment_type(env, arg_string):
    """
    Convert, THTensor, THStorage, THLongTensor, etc. to a tuple of:
    1. the "Container" - e.g. Tensor or Storage
    2. the processor - e.g. CPU, or CUDA, from env
    3. the "Scalar Type" - e.g. long, int, from the arg_string or env
    in the case it is generic
    """
    # processor is always taken from env
    processor = env['Processor']

    # scalar name might be taken from env
    scalar_fallback = env['ScalarName']

    # Attempt to match on THTensor
    tensor_re = re.compile(r"TH([a-zA-Z]*)Tensor\*")
    match = tensor_re.match(arg_string)
    if match is not None:
        # if the Tensor has a type specified, use it
        scalar_type = (match.group(1) if len(match.group(1)) > 0 else
                       scalar_fallback)
        return ('Tensor', processor, scalar_type)

    # Attempt to match on THStorage
    storage_re = re.compile(r"TH([a-zA-Z]*)Storage\*")
    match = storage_re.match(arg_string)
    if match is not None:
        # if the Storage has a type specified, use it
        scalar_type = (match.group(1) if len(match.group(1)) > 0 else
                       scalar_fallback)
        return ('Storage', processor, scalar_type)

    # Handle THSize? also should this be Long or int64_t
    if arg_string == 'THSize*':
        return ('Storage', 'CPU', 'Long')

    # for now, just return None, up to you what the failure case is
    return None


cpu_env = {
    'Processor': 'CPU',
    'ScalarName': 'Float',
}

cuda_env = {
    'Processor': 'CUDA',
    'ScalarName': 'Half',
}

print(to_environment_type(cpu_env, 'THTensor*'))
print(to_environment_type(cpu_env, 'THLongStorage*'))
print(to_environment_type(cuda_env, 'THIntTensor*'))
print(to_environment_type(cuda_env, 'THSize*'))
