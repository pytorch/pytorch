def __getattr__(name):
    return f"<dynamic:{name}>"


def create_delete_missing_value_fn():
    global create_delete_missing_value
    create_delete_missing_value = 5
    del create_delete_missing_value
