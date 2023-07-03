## @package formatter
# Module caffe2.python.docs.formatter




from caffe2.python.docs.parser import Parser


class Formatter:
    def __init__(self):
        self.content = ""

    def clone(self):
        return self.__class__()

    def dump(self):
        return self.content

    def parseAndAdd(self, text):
        text = Parser(text, self).parse()
        self.addRaw(text)

    def addRaw(self, text):
        raise Exception('Not yet implemented.')

    def addLine(self, text):
        raise Exception('Not yet implemented.')

    def addLinebreak(self):
        raise Exception('Not yet implemented.')

    def addHeader(self, text):
        raise Exception('Not yet implemented.')

    def addEmphasis(self, text):
        raise Exception('Not yet implemented.')

    def addList(self, textList):
        raise Exception('Not yet implemented.')

    def addLink(self, text, url):
        raise Exception('Not yet implemented.')

    def addCode(self, text):
        raise Exception('Not yet implemented.')

    def addCodeLink(self, text):
        raise Exception('Not yet implemented.')

    def addTable(self, table):
        raise Exception('Not yet implemented.')

    def addBreak(self):
        raise Exception('Not yet implemented.')


class Markdown(Formatter):
    def addRaw(self, text):
        self.content += "{text}".format(text=text)

    def addLine(self, text, new_line=False):
        self.content += "{line}{text}\n".format(line=('\n' if new_line else ''),
                                                text=text)

    def addLinebreak(self):
        self.content += "\n"

    def addHeader(self, text, h=1):
        self.addLine("{header} {text}".format(header=h * '#', text=text), True)

    def addEmphasis(self, text, s=1):
        self.addRaw("{stars}{text}{stars}".format(stars=s * '*', text=text))

    def addList(self, textList):
        for text in textList:
            self.addLine("- {text}".format(text=text), True)
        self.addLinebreak()

    def addLink(self, text, url):
        self.addRaw("[{text}]({url})".format(text=text, url=url))

    def addCodeLink(self, path, options=None):
        self.addRaw("({path})".format(path=path))

    def addCode(self, text, inline=False):
        if (inline):
            self.content += "`{text}`".format(text=text)
        else:
            self.addRaw("\n\n```\n{text}```\n\n".format(text=text))

    def addTable(self, table, noTitle=False):
        self.addLinebreak()
        assert(len(table) > 1)
        if noTitle:
            table.insert(0, [' ' for i in range(len(table[0]))])
        self.addLine(' | '.join(table[0]))
        self.addLine(' | '.join(['----' for i in range(len(table[0]))]))
        for row in table[1:]:
            self.addLine(' | '.join(row))
        self.addLinebreak()

    def addBreak(self):
        self.addLine('\n---\n', True)
