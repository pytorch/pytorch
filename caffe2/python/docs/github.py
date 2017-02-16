from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.docs.formatter import Markdown
from caffe2.python.docs.generator import OpDocGenerator, DocUploader
from caffe2.python.docs.generator import OperatorDoc, OperatorEngine
import os


class GHOpDocUploader(DocUploader):
    def __init__(self):
        pass

    def upload(self, content_body):
        print(content_body)


class GHMarkdown(Markdown):
    def addHeader(self, text, h=1):
        if h == 2:
            h = 1
        self.addLine("{header} {text}".format(header=h * '#', text=text), True)

    def addTable(self, table, noTitle=False):
        self.addRaw("<table>")
        for row in table:
            self.addRaw("<tr>")
            for cell in row:
                self.addRaw("<td>")
                self.addLine("{cell}".format(cell=cell))
                self.addRaw("</td>")
            self.addRaw("</tr>")
        self.addRaw("</table>")


def getCodeLink(formatter, schema):
    formatter = formatter.clone()
    path = os.path.relpath(schema.file, "caffe2")
    schemaLink = ('https://github.com/caffe2/caffe2/blob/master/{path}'
                  .format(path=path))
    formatter.addLink('{path}'.format(path=path), schemaLink)
    return formatter.dump()


class GHOperatorEngine(OperatorEngine):
    def generateDoc(self, formatter, schema):
        for device, _ in self.getDeviceImpl():
            formatter.addCode('{engine}'.format(engine=self.engine), True)
            if device:
                formatter.addRaw(' on ')
                formatter.addEmphasis("{device}".format(device=device), 1)


class GHOperatorDoc(OperatorDoc):
    def generateCodeLink(self, formatter):
        formatter.addHeader("Code", 3)
        formatter.addRaw(getCodeLink(formatter, self.schema))

    def getInfo(self, formatter, name, impl):
        formatter = formatter.clone()
        if impl:
            formatter.addEmphasis('{name}'.format(name=name), 1)
            formatter.addRaw(' ')
            formatter.addCode('{impl}'.format(impl=impl), True)
        return formatter.dump()


class GHOpDocGenerator(OpDocGenerator):
    def getOperatorDoc(self, name, schema, priority):
        return GHOperatorDoc(name, schema, priority)

    def getOperatorEngine(self, name):
        return GHOperatorEngine(name)


if __name__ == "__main__":
    ops = GHOpDocGenerator(GHMarkdown())
    ops.createBody()
    print(ops.content_body)
