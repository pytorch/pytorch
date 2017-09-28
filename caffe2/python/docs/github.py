# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package github
# Module caffe2.python.docs.github
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
        self.addLine("\n{header} {text}\n".format(header=h * '#', text=text), True)

    def addDocHeader(self):
        self.addLine("---")
        self.addLine("docid: operators-catalogue")
        self.addLine("title: Operators Catalogue")
        self.addLine("layout: docs")
        self.addLine("permalink: /docs/operators-catalogue.html")
        self.addLine("---")
        self.addLine("* TOC")
        self.addLine("{:toc}")

    def addTable(self, table, noTitle=False):
        self.addLinebreak()
        assert(len(table) > 1)
        self.addLine(' | '.join(['----------' for i in range(len(table[0]))]))
        self.addLine(' | '.join(table[0]))
        for row in table[1:]:
            self.addLine(' | '.join(row))

    def addTableHTML(self, table, noTitle=False):
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
    def generateDoc(self, formatter):
        for device, _ in self.getDeviceImpl():
            formatter.addCode('{engine}'.format(engine=self.engine), True)
            if device:
                formatter.addRaw(' on ')
                formatter.addEmphasis("{device}".format(device=device), 1)


class GHOperatorDoc(OperatorDoc):
    def generateCodeLink(self, formatter):
        formatter.addHeader("Code", 3)
        formatter.addLinebreak()
        formatter.addRaw(getCodeLink(formatter, self.schema))

    def getInfo(self, formatter, name, impl):
        formatter = formatter.clone()
        if impl:
            formatter.addEmphasis('{name}'.format(name=name), 1)
            formatter.addRaw(' ')
            formatter.addCode('{impl}'.format(impl=impl), True)
        return formatter.dump()

    def generateSchema(self, formatter):
        formatter.addHeader(self.name, 2)
        if self.schema:
            self.generateDoc(formatter)
            self.generateInterface(formatter)
            self.generateCodeLink(formatter)
            formatter.addBreak()
        else:
            formatter.addLine("No schema documented yet.")


class GHOpDocGenerator(OpDocGenerator):
    def getOperatorDoc(self, name, schema, priority):
        return GHOperatorDoc(name, schema, priority)

    def getOperatorEngine(self, name):
        return GHOperatorEngine(name)

    def createBody(self):
        self.formatter.addDocHeader()
        operators = self.getOperators()

        for operator in operators:
            operator.generateSchema(self.formatter)

        self.content_body += self.formatter.dump()


if __name__ == "__main__":
    ops = GHOpDocGenerator(GHMarkdown(), GHOpDocUploader)
    ops.createBody()
    print(ops.content_body)
