import inspect
import pickle


# Construct functions that save/load the state of the config module `module`.
# The config settings are expected to either be module-level globals or
# class variables.
# `ignore_set` is a set of names of configurations to ignore. e.g. if you
# want to ignore config.x and config.y.z in your config module, then
# `ignore_set` should be {"x", "y.z"}.
def get_config_serialization_fns(module, ignore_set=None):
    def _save(obj, name_prefix):
        saved_state = {}
        for key, val in obj.__dict__.items():
            if ignore_set is not None and name_prefix + key in ignore_set:
                continue
            try:
                pickle.dumps(val)
            except Exception:
                pass
            else:
                saved_state[key] = (
                    _save(val, name_prefix + key + ".") if inspect.isclass(val) else val
                )
        return saved_state

    def save_config():
        return pickle.dumps(_save(module, ""))

    def _load(obj, data):
        for key, val in data.items():
            attr = getattr(obj, key, None)
            if attr is not None and inspect.isclass(attr):
                _load(attr, val)
            else:
                try:
                    setattr(obj, key, val)
                except Exception:
                    pass

    def load_config(data):
        _load(module, pickle.loads(data))

    return save_config, load_config
