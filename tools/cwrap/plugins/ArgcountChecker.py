from . import CWrapPlugin


class ArgcountChecker(CWrapPlugin):

    def process_all_checks(self, checks, option):
        if not checks:
            checks = '__argcount == 0'
        else:
            indent = '\n          '
            argcount = option['num_checked_args'] + option.get('argcount_offset', 0)
            checks = '__argcount == {} &&'.format(str(argcount)) + indent + checks
        return checks
