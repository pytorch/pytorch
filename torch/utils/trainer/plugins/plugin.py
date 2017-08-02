
class Plugin(object):

    def __init__(self, interval=None):
        """
            Args:
                interval: A list, e.g. [(10, 'iteration'), (1, 'epoch')] which 
                    specifies that the plugin should be called every 10 iterations
                    and also every epoch.
        """
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, trainer):
        raise NotImplementedError


class PluginFactory(Plugin):

    def __init__(self, fn, register_fn=None, interval=None):
        """ 
            Creates a Plugin which applies fn at each hook specified in 'interval' 
        """
        super(PluginFactory, self).__init__(interval)
        for _, name in interval:
            setattr(self, name, fn)

        if not register_fn:
            def register(self, trainer):
                self.trainer = trainer
            self.register = register
