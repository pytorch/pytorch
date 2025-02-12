from concurrent.futures import Future


def wrap_future(original_future: Future) -> Future[None]:
    masked_future = Future()

    def on_complete(future):
        try:
            original_future.result()
            masked_future.set_result(None)
        except Exception as e:
            # TODO dont mess up the stack trace
            masked_future.set_exception(e)

    original_future.add_done_callback(on_complete)
    return masked_future


def get_state_dict_fqns(state_dict: dict[str, Any]) -> list[str]:
    """
    Returns a list of fully qualified names (FQNs) for the given state_dict.

    Args:
        state_dict (dict[str, Any]): The state_dict to get the FQNs for.

    Returns:
        list[str]: A list of FQNs for the given state_dict.
    """
    pass
    return []
