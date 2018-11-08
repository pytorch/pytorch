## @package generator
# Module caffe2.python.docs.generator
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
from caffe2.python import core, workspace
from caffe2.python.docs.formatter import Markdown
from future.utils import viewitems, viewvalues

OpSchema = workspace.C.OpSchema


class DocUploader(object):
    def __init__(self):
        pass

    def upload(self, text):
        pass


class DocGenerator(object):
    def __init__(self, formatter, uploader):
        self.formatter = formatter
        self.uploader = uploader
        self.content_body = ""

    def create_body(self):
        pass

    def update(self):
        self.uploader.upload(self.content_body)


class OpDocGenerator(DocGenerator):
    def getOperatorDoc(self, name, schema, priority):
        return OperatorDoc(name, schema, priority)

    def getOperatorEngine(self, name):
        return OperatorEngine(name)

    def getOperators(self):
        # map: op_name -> operator
        self.operators = {}
        # map: op_name -> [engine, engine]
        self.engines = {}

        def filePriority(x):
            if x == "caffe2/caffe2/operators":
                return 0
            if 'contrib' in x.split('/'):
                return 2
            if 'experiments' in x.split('/'):
                return 3
            return 1

        for name in core._GetRegisteredOperators():
            schema = OpSchema.get(name)
            if schema:
                priority = filePriority(os.path.dirname(schema.file))
                operator = self.getOperatorDoc(name, schema, priority)
                self.operators[name] = operator

            # Engine
            elif name.find("_ENGINE_") != -1:
                engine = self.getOperatorEngine(name)
                if engine.base_op_name in self.engines:
                    self.engines[engine.base_op_name].append(engine)
                else:
                    self.engines[engine.base_op_name] = [engine]

            # No schema
            else:
                priority = 4
                self.operators[name] = self.getOperatorDoc(name, schema, priority)

        for name, engines in viewitems(self.engines):
            if name in self.operators:
                self.operators[name].addEngines(engines)

        # Generate a sorted list of operators
        return sorted(
            viewvalues(self.operators),
            key=lambda op: (op.priority, op.name)
        )

    def createBody(self):
        operators = self.getOperators()

        for operator in operators:
            operator.generateSchema(self.formatter)

        self.content_body += self.formatter.dump()


class OperatorEngine(object):
    def __init__(self, name):
        self.op_name = name
        self.base_op_name, self.engine = name.split("_ENGINE_", 1)

    def getDeviceImpl(self):
        deviceImplList = []
        for device, impl in [('CPU', OpSchema.get_cpu_impl(self.op_name)),
                             ('CUDA', OpSchema.get_cuda_impl(self.op_name))]:
            if not impl:
                continue
            deviceImplList.append((device, impl))
        return deviceImplList

    def generateDoc(self, formatter):
        for device, impl in self.getDeviceImpl():
            formatter.addLine(
                '{engine} on {device}: {impl}'.format(engine=self.engine,
                                                      device=device,
                                                      impl=impl))


class OperatorDoc(object):
    def __init__(self, name, schema, priority):
        self.name = name
        self.schema = schema
        self.priority = priority
        print("Gathering docs for {}...".format(self.name))
        self.engines = []

    def addEngines(self, engines):
        self.engines = engines

    def generateDoc(self, formatter):
        if self.schema.doc:
            formatter.parseAndAdd(self.schema.doc)
            formatter.addLinebreak()
        else:
            formatter.addLine("No documentation yet.")

    def generateTable(self, formatter, tuples, title_row, title):
        if tuples:
            if title:
                formatter.addHeader(title, 3)
            table = []
            if title_row:
                table = [title_row]
            for name, doc in tuples:
                table.append([name, doc or ''])
            formatter.addTable(table, (table == []))

    def generateInterface(self, formatter):
        def makeDesc(title, args):
            f = formatter.clone()
            f.addEmphasis(title, 1)
            out = [(f.dump(), '')]
            for arg in args:
                f = formatter.clone()
                if isinstance(arg, tuple):
                    name = arg[0]
                    if len(arg) > 1:
                        description = arg[1] or ''
                    else:
                        description = ''
                else:
                    name = arg.name
                    description = arg.description or ''
                f.addCode(name, inline=True)
                out.append((f.dump(), description or ''))
            return out

        tuples = []

        if self.schema.args:
            tuples += makeDesc('Arguments', self.schema.args)

        if self.schema.input_desc:
            tuples += makeDesc('Inputs', self.schema.input_desc)

        if self.schema.output_desc:
            tuples += makeDesc('Outputs', self.schema.output_desc)

        self.generateTable(formatter, tuples, None, 'Interface')
        print("Generated interface for {}".format(self.name))

    def generateCodeLink(self, formatter):
        formatter.addHeader("Code", 3)
        formatter.addLinebreak()
        formatter.addCodeLink(self.schema.file)

    def getInfo(self, formatter, name, impl):
        pass

    def generateDevices(self, formatter):
        formatter.addHeader("Devices", 3)
        devices = [
            self.getInfo(formatter,
                         'CPU', OpSchema.get_cpu_impl(self.name)),
            self.getInfo(formatter,
                         'GPU', OpSchema.get_cuda_impl(self.name)),
        ]
        formatter.addList([i for i in devices if i])

    def generateEngines(self, formatter):
        if not len(self.engines):
            return
        formatter.addHeader("Engines", 3)
        for engine in self.engines:
            engine.generateDoc(formatter)

    def generateSchema(self, formatter):
        formatter.addHeader(self.name, 2)
        if self.schema:
            self.generateDoc(formatter)
            self.generateInterface(formatter)
            self.generateCodeLink(formatter)
            self.generateDevices(formatter)
            self.generateEngines(formatter)
            formatter.addBreak()
        else:
            formatter.addLine("No schema documented yet.")
            self.generateDevices(formatter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operators catalog generator.")
    parser.add_argument('catalog_path', type=str,
                        help='operators-catalogue.md to write out to')
    args = parser.parse_args()

    with open(args.catalog_path, 'w') as fp:
        ops = OpDocGenerator(Markdown(), DocUploader())
        ops.createBody()
        fp.write(ops.content_body)
