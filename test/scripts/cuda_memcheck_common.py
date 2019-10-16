class Report:
    HEAD = 'ERROR SUMMARY: '
    TAIL = ' errors'

    def __init__(self, text, errors):
        self.text = text
        self.num_errors = int(text[len(self.HEAD):len(text) - len(self.TAIL)])
        self.errors = errors
        assert len(errors) == self.num_errors


class Error:

    def __init__(self, lines):
        self.message = lines[0]
        lines = lines[2:]
        self.stack = [l.strip() for l in lines]


def parse(message):
    errors = []
    HEAD = '========='
    headlen = len(HEAD)
    started = False
    in_message = False
    message_lines = []
    lines = message.splitlines()
    for l in lines:
        if l == HEAD + ' CUDA-MEMCHECK':
            started = True
            continue
        if not started or not l.startswith(HEAD):
            continue
        l = l[headlen + 1:]
        if l.startswith('ERROR SUMMARY:'):
            return Report(l, errors)
        if not in_message:
            assert l[0] != ' '
            in_message = True
            message_lines = [l]
        elif l == '':
            errors.append(Error(message_lines))
            in_message = False
        else:
            assert l[0] == ' '
            message_lines.append(l)
