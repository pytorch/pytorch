import inspect
import pickle


def get_config_serialization_fns(module):
    def _save(obj):
        saved_state = {}
        for key, val in obj.__dict__.items():
            try:
                pickle.dumps(val)
            except Exception:
                pass
            else:
                saved_state[key] = _save(val) if inspect.isclass(val) else val
        return saved_state

    def save_config():
        return pickle.dumps(_save(module))

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
