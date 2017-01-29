from . import CWrapPlugin


class ArgcountSortPlugin(CWrapPlugin):

    def __init__(self, descending=True):
        self.descending = descending

    def process_declarations(self, declarations):
        def num_checked_args(option):
            return sum(map(lambda a: not a.get('ignore_check', False), option['arguments']))
        for declaration in declarations:
            declaration['options'].sort(key=num_checked_args, reverse=self.descending)
        return declarations
