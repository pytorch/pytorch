NUMERIC_CONSTANT = 42
STRING_CONSTANT = "Alice"
LIST_CONSTANT = [1, 2, 3, 4, 5]
DICT_CONSTANT = {"key1": "value1", "key2": "value2"}
TUPLE_CONSTANT = (1, 2, 3)
SET_CONSTANT = {1, 2, 3, 4, 5}
BOOLEAN_CONSTANT = True
NONE_CONSTANT = None


def greeter(name=None):
    return f"Hello, {name}!"


greeter(STRING_CONSTANT)
greeter(NUMERIC_CONSTANT)
