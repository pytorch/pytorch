tensor_options_optional_types_and_var = ['c10::optional<ScalarType> dtype', 'c10::optional<Layout> layout', 'c10::optional<Device> device', 'c10::optional<bool> pin_memory']
tensor_options_types_and_var = ['ScalarType dtype', 'Layout layout', 'Device device', 'bool pin_memory']

def check_if_factory_method(args):
    for arg in args: 
        if 'type' not in arg:
            return False

    a = any(arg['type'] == 'c10::optional<ScalarType>' for arg in args) and any(arg['type'] == 'c10::optional<Layout>' for arg in args) and any(arg['type'] == 'c10::optional<Device>' for arg in args) and any(arg['type'] == 'c10::optional<bool>' for arg in args)
    c = any(arg['type'] == 'ScalarType' for arg in args) and any(arg['type'] == 'Layout' for arg in args) and any(arg['type'] == 'Device' for arg in args) and any(arg['type'] == 'bool' for arg in args)
    b = any('TensorOptions' in arg['type'] for arg in args)

    return a or b or c

def collapse_actuals2(actuals):
    collapsed = actuals[:]
    index = actuals.index('dtype')
    collapsed[index] = 'at::typeMetaToScalarType(options.dtype())'
    collapsed[index + 1] = 'options.layout()'
    collapsed[index + 2] = 'options.device()'
    collapsed[index + 3] = 'options.pinned_memory()'
    return collapsed


def check_tensor_options_in_formals(formals):
    return (any(formal['dynamic_type'] == 'ScalarType' for formal in formals) and
            any(formal['dynamic_type'] == 'Layout' for formal in formals) and
            any(formal['dynamic_type'] == 'Device' for formal in formals) and 
            any(formal['dynamic_type'] == 'bool' for formal in formals))

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

def collapse_formals(formals):
    collapsed = formals[:]

    hasTO = True
    hasDefVal = False
    for option in tensor_options_optional_types_and_var:
        if not any(option in formal for formal in formals):
            hasTO = False
            break

        if any((option in formal and '=' in formal) for formal in formals) :
            hasDefVal = True

        if hasTO:
            index = [idx for idx, formal in enumerate(formals) if 'c10::optional<ScalarType> dtype' in formal][0]
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            if hasDefVal:
                collapsed.insert(index, 'const at::TensorOptions & options={}')
            else:
                collapsed.insert(index, 'const at::TensorOptions & options')

            return collapsed

    hasTO = True
    for option in tensor_options_types_and_var:
        if not any(option in formal for formal in formals):
            hasTO = False
            break

        if any((option in formal and '=' in formal) for formal in formals) :
            hasDefVal = True

        if hasTO:
            index = [idx for idx, formal in enumerate(formals) if 'ScalarType dtype' in formal][0]
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            if hasDefVal:
                collapsed.insert(index, 'const at::TensorOptions & options={}')
            else:
                collapsed.insert(index, 'const at::TensorOptions & options')

            return collapsed

    return collapsed

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
            collapsed.insert(index, {"annotation" : "None", "dynamic_type": "TensorOptions", "is_nullable": "False", "default": "{}", "kwarg_only": "True", "name": "options", "type": "const TensorOptions &", })

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
            collapsed.insert(index, {"annotation" : "None", "dynamic_type": "TensorOptions", "is_nullable": "False", "kwarg_only": "True", "name": "options", "type": "const TensorOptions &", })

        return collapsed
