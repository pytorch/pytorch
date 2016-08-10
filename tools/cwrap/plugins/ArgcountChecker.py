from . import CWrapPlugin

class ArgcountChecker(CWrapPlugin):

    def process_all_checks(self, checks, option):
        if not checks:
            checks = '__argcount == 0'
        else:
            indent = '\n          '
            checks = '__argcount == {} &&'.format(option['num_checked_args']) + \
                indent + checks
        return checks
