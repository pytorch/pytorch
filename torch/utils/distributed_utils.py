from typing import Dict, Any, Union


def remove_prefix_from_state_dict_if_exists(state_dict: Union[Dict[str, Any], Any],
                                            prefix: str) -> Union[Dict[str, Any], Any]:
    """This function strips a prefix if it exists in the model
    state_dict and metadata.

    Args:
        state_dict : DP/DDP pytorch model state_dict
        prefix (str) : Prefix to be removed from the DP/DDP model state_dict to
                       to make it compatible with regular pytorch model.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in list(state_dict.keys()):
        new_key = key[len(prefix):]
        state_dict[new_key] = state_dict.pop(key)

    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass

    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)
    return state_dict
