
class Plugin(object):

    def __init__(self, interval=None):
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, trainer):
        raise NotImplementedError
