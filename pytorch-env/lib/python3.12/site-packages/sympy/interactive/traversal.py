from sympy.core.basic import Basic
from sympy.printing import pprint

import random

def interactive_traversal(expr):
    """Traverse a tree asking a user which branch to choose. """

    RED, BRED = '\033[0;31m', '\033[1;31m'
    GREEN, BGREEN = '\033[0;32m', '\033[1;32m'
    YELLOW, BYELLOW = '\033[0;33m', '\033[1;33m'  # noqa
    BLUE, BBLUE = '\033[0;34m', '\033[1;34m'      # noqa
    MAGENTA, BMAGENTA = '\033[0;35m', '\033[1;35m'# noqa
    CYAN, BCYAN = '\033[0;36m', '\033[1;36m'      # noqa
    END = '\033[0m'

    def cprint(*args):
        print("".join(map(str, args)) + END)

    def _interactive_traversal(expr, stage):
        if stage > 0:
            print()

        cprint("Current expression (stage ", BYELLOW, stage, END, "):")
        print(BCYAN)
        pprint(expr)
        print(END)

        if isinstance(expr, Basic):
            if expr.is_Add:
                args = expr.as_ordered_terms()
            elif expr.is_Mul:
                args = expr.as_ordered_factors()
            else:
                args = expr.args
        elif hasattr(expr, "__iter__"):
            args = list(expr)
        else:
            return expr

        n_args = len(args)

        if not n_args:
            return expr

        for i, arg in enumerate(args):
            cprint(GREEN, "[", BGREEN, i, GREEN, "] ", BLUE, type(arg), END)
            pprint(arg)
            print()

        if n_args == 1:
            choices = '0'
        else:
            choices = '0-%d' % (n_args - 1)

        try:
            choice = input("Your choice [%s,f,l,r,d,?]: " % choices)
        except EOFError:
            result = expr
            print()
        else:
            if choice == '?':
                cprint(RED, "%s - select subexpression with the given index" %
                       choices)
                cprint(RED, "f - select the first subexpression")
                cprint(RED, "l - select the last subexpression")
                cprint(RED, "r - select a random subexpression")
                cprint(RED, "d - done\n")

                result = _interactive_traversal(expr, stage)
            elif choice in ('d', ''):
                result = expr
            elif choice == 'f':
                result = _interactive_traversal(args[0], stage + 1)
            elif choice == 'l':
                result = _interactive_traversal(args[-1], stage + 1)
            elif choice == 'r':
                result = _interactive_traversal(random.choice(args), stage + 1)
            else:
                try:
                    choice = int(choice)
                except ValueError:
                    cprint(BRED,
                           "Choice must be a number in %s range\n" % choices)
                    result = _interactive_traversal(expr, stage)
                else:
                    if choice < 0 or choice >= n_args:
                        cprint(BRED, "Choice must be in %s range\n" % choices)
                        result = _interactive_traversal(expr, stage)
                    else:
                        result = _interactive_traversal(args[choice], stage + 1)

        return result

    return _interactive_traversal(expr, 0)
