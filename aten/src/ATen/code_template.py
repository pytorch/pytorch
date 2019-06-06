import re

# match $identifier or ${identifier} and replace with value in env
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block subsitution by indenting to that depth and putting each element
# of the list on its own line
# if the identifier is on a line starting with non-whitespace and a list
# then it is comma separated ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.


class CodeTemplate(object):
    # Python 2.7.5 has a bug where the leading (^[^\n\S]*)? does not work,
    # workaround via appending another [^\n\S]? inside

    substitution_str = r'(^[^\n\S]*[^\n\S]?)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})'

    # older versions of Python have a bug where \w* does not work,
    # so we need to replace with the non-shortened version [a-zA-Z0-9_]*
    # https://bugs.python.org/issue18647

    substitution_str = substitution_str.replace(r'\w', r'[a-zA-Z0-9_]')

    subtitution = re.compile(substitution_str, re.MULTILINE)

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern, filename=""):
        self.pattern = pattern
        self.filename = filename

    def substitute(self, env=None, **kwargs):
        if env is None:
            env = {}

        def lookup(v):
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent, v):
            return "".join([indent + l + "\n" for e in v for l in str(e).splitlines()]).rstrip()

        def replace(match):
            indent = match.group(1)
            key = match.group(2)
            comma_before = ''
            comma_after = ''
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ', '
                    key = key[1:]
                if key[-1] == ',':
                    comma_after = ', '
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ', '.join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)
        return self.subtitution.sub(replace, self.pattern)


if __name__ == "__main__":
    c = CodeTemplate("""\
    int foo($args) {

        $bar
            $bar
        $a+$b
    }
    int commatest(int a${,stuff})
    int notest(int a${,empty,})
    """)
    print(c.substitute(args=["hi", 8], bar=["what", 7],
                       a=3, b=4, stuff=["things...", "others"], empty=[]))
