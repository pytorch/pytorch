# These are helper methods around TensorOptions object in codegen
# While processing native_funtionc.yaml we heavily rely on the parameters types and orders
# TensorOptions object consists of ScalarType, Layout, Device and Bool (pin_memory) types

# List of tensor options
tensor_options_args = ['dtype', 'layout', 'device', 'pin_memory']

tensor_options_optional_types_and_var = ['c10::optional<ScalarType> dtype',
                                         'c10::optional<Layout> layout',
                                         'c10::optional<Device> device',
                                         'c10::optional<bool> pin_memory']

tensor_options_types_and_var = ['ScalarType dtype', 'Layout layout', 'Device device', 'bool pin_memory']

# Checks if passed list of arguments contains TensorOptions in it.
# Returns 'True' in cases when:
#   1. There is 'TensorOptions' type argument
#   2. There are all required types of arguments
#   3. There are all required types of arguments as optional
def check_if_factory_method(args):
    for arg in args:
        if 'type' not in arg:
            return False

    has_opt_TO_args = any(arg['type'] == 'c10::optional<ScalarType>' for arg in args) and \
        any(arg['type'] == 'c10::optional<Layout>' for arg in args) and \
        any(arg['type'] == 'c10::optional<Device>' for arg in args) and \
        any(arg['type'] == 'c10::optional<bool>' for arg in args)

    has_TO_args = any(arg['type'] == 'ScalarType' for arg in args) and \
        any(arg['type'] == 'Layout' for arg in args) and \
        any(arg['type'] == 'Device' for arg in args) and \
        any(arg['type'] == 'bool' for arg in args)

    has_TO_arg = any('TensorOptions' in arg['type'] for arg in args)

    return has_opt_TO_args or has_TO_args or has_TO_arg

# Checks if passed formals have TensorOption arguments in it
def check_tensor_options_in_formals(formals):
    return (any(formal['dynamic_type'] == 'ScalarType' for formal in formals) and
            any(formal['dynamic_type'] == 'Layout' for formal in formals) and
            any(formal['dynamic_type'] == 'Device' for formal in formals) and
            any(formal['dynamic_type'] == 'bool' for formal in formals))

# Find 'dtype' in actuals, remove it and 3 next elements and insert 'options'
# instead. This method relies on a strict order of TensorOption arguments.
def collapse_actuals(actuals):
    collapsed = actuals[:]
    if (any(actual == 'dtype' for actual in actuals) and
        any(actual == 'layout' for actual in actuals) and
        any(actual == 'device' for actual in actuals) and
            any(actual == 'pin_memory' for actual in actuals)):
        index = collapsed.index('dtype')

        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.insert(index, 'options')

    return collapsed

# Collapses tensor options formals into one TensorOptions formal argument
# Covers cases for optional and non-optional tensor options arguments
def collapse_formals(formals):
    collapsed = formals[:]

    has_TO = True
    has_def_val = False
    for option in tensor_options_optional_types_and_var:
        if not any(option in formal for formal in formals):
            has_TO = False
            break

        if any((option in formal and '=' in formal) for formal in formals) :
            has_def_val = True

        if has_TO:
            index = [idx for idx, formal in enumerate(formals) if 'c10::optional<ScalarType> dtype' in formal][0]
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            if has_def_val:
                collapsed.insert(index, 'const at::TensorOptions & options={}')
            else:
                collapsed.insert(index, 'const at::TensorOptions & options')

            return collapsed

    has_TO = True
    for option in tensor_options_types_and_var:
        if not any(option in formal for formal in formals):
            has_TO = False
            break

        if any((option in formal and '=' in formal) for formal in formals) :
            has_def_val = True

        if has_TO:
            index = [idx for idx, formal in enumerate(formals) if 'ScalarType dtype' in formal][0]
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            if has_def_val:
                collapsed.insert(index, 'const at::TensorOptions & options={}')
            else:
                collapsed.insert(index, 'const at::TensorOptions & options')

            return collapsed

    return collapsed

# Collapses tensor options formals into one TensorOptions formal argument object
# with all the metadata.
# Covers cases for optional and non-optional tensor options arguments
def collapse_formals_list(formals):
    collapsed = formals[:]
    if (any(formal['type'] == 'c10::optional<ScalarType>' for formal in collapsed) and
        any(formal['type'] == 'c10::optional<Layout>' for formal in collapsed) and
        any(formal['type'] == 'c10::optional<Device>' for formal in collapsed) and
            any(formal['type'] == 'c10::optional<bool>' for formal in collapsed)):
        index = 0
        for i in range(len(collapsed)):
            if collapsed[i]['type'] == 'c10::optional<ScalarType>':
                break
            else:
                index += 1

        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.insert(index, {"annotation" : "None",
                                 "dynamic_type": "TensorOptions",
                                 "is_nullable": "False",
                                 "default": "{}",
                                 "kwarg_only": "True",
                                 "name": "options",
                                 "type": "const TensorOptions &", })

    if (any(formal['type'] == 'ScalarType' for formal in collapsed) and
        any(formal['type'] == 'Layout' for formal in collapsed) and
        any(formal['type'] == 'Device' for formal in collapsed) and
            any(formal['type'] == 'bool' for formal in collapsed)):
        index = 0
        for i in range(len(collapsed)):
            if collapsed[i]['type'] == 'ScalarType':
                break
            else:
                index += 1

        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.pop(index)
        collapsed.insert(index, {"annotation" : "None",
                                 "dynamic_type": "TensorOptions",
                                 "is_nullable": "False",
                                 "kwarg_only": "True",
                                 "name": "options",
                                 "type": "const TensorOptions &", })

    return collapsed


def check_hack(name):
    return name in ['randint_like', 'rand_like', 'randn_like', 'zeros_like', 'ones_like', 'full_like', 'empty_like', '_cudnn_init_dropout_state', '_sparse_coo_tensor_with_dims_and_tensors']
