## @package parser
# Module caffe2.python.docs.parser
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re


class Parser(object):
    # List of tuples (regex_str, lambda(regex_match, formatter))
    # If a lambda returns True it will be called repeatedly with replacement
    # otherwise it will only be called on text that hasn't been parsed yet.
    regexes = [
        # Code blocks of various formats
        ('````(.+?)````',
         lambda m, f: f.addCode(m.group(1))
         ),
        ('```(.+?)```',
         lambda m, f: f.addCode(m.group(1))
         ),
        ('((( {2})+)(\S.*)(\n\s*\n|\n))+',
         lambda m, f: f.addCode(m.group(0))
         ),
        ('([^\.])\n',
         lambda m, f: f.addRaw('{c} '.format(c=m.group(1))) or True
         ),
        ('`(.+?)`',
         lambda m, f: f.addCode(m.group(1), True)
         ),
        # Make links clickable
        ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]'
         '|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
         lambda m, f: f.addLink(m.group(0), m.group(0))
         ),
        ('\*\*(.+?)\*\*',
         lambda m, f: f.addEmphasis(m.group(1), 2)
         ),
        ('\*(.+?)\*',
         lambda m, f: f.addEmphasis(m.group(1), 1)
         ),
    ]

    def __init__(self, text, formatter):
        self.text = text
        self.lines = []
        self.formatter = formatter

    def parseText(self):
        UNPARSED = 0
        PARSED = 1
        parsed_block = [(UNPARSED, self.text)]
        for regex, func in self.regexes:
            index = 0
            while index < len(parsed_block):
                label, text = parsed_block[index]

                # Already been parsed
                if (label == PARSED):
                    index += 1
                    continue

                match = re.search(regex, text)
                if match:
                    parsed_block.pop(index)
                    start = match.start(0)
                    end = match.end(0)

                    f = self.formatter.clone()
                    merge = func(match, f)

                    if merge:
                        merged = text[:start] + f.dump() + text[end:]
                        parsed_block.insert(index, (UNPARSED, merged))
                    else:
                        if text[:start]:
                            parsed_block.insert(index,
                                                (UNPARSED, text[:start]))

                        index += 1
                        parsed_block.insert(index, (PARSED, f.dump()))

                        index += 1
                        if text[end:]:
                            parsed_block.insert(index,
                                                (UNPARSED, text[end:]))

                else:
                    index += 1

        self.lines += [i for _, i in parsed_block]
        self.text = ' '.join(self.lines)

    def parse(self):
        self.parseText()
        return self.text
