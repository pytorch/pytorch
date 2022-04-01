




from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import os
import tempfile
from zipfile import ZipFile

'''
Generates a document in markdown format summrizing the coverage of serialized
testing. The document lives in
`caffe2/python/serialized_test/SerializedTestCoverage.md`
'''

OpSchema = workspace.C.OpSchema


def gen_serialized_test_coverage(source_dir, output_dir):
    (covered, not_covered, schemaless) = gen_coverage_sets(source_dir)
    num_covered = len(covered)
    num_not_covered = len(not_covered)
    num_schemaless = len(schemaless)
    total_ops = num_covered + num_not_covered

    with open(os.path.join(output_dir, 'SerializedTestCoverage.md'), 'w+') as f:
        f.write('# Serialized Test Coverage Report\n')
        f.write("This is an automatically generated file. Please see "
            "`caffe2/python/serialized_test/README.md` for details. "
            "In the case of merge conflicts, please rebase and regenerate.\n")
        f.write('## Summary\n')
        f.write(
            'Serialized tests have covered {}/{} ({}%) operators\n\n'.format(
                num_covered, total_ops,
                (int)(num_covered / total_ops * 1000) / 10))

        f.write('## Not covered operators\n')
        f.write('<details>\n')
        f.write(
            '<summary>There are {} not covered operators</summary>\n\n'.format(
                num_not_covered))
        for n in sorted(not_covered):
            f.write('* ' + n + '\n')
        f.write('</details>\n\n')

        f.write('## Covered operators\n')
        f.write('<details>\n')
        f.write(
            '<summary>There are {} covered operators</summary>\n\n'.format(
                num_covered))
        for n in sorted(covered):
            f.write('* ' + n + '\n')
        f.write('</details>\n\n')

        f.write('## Excluded from coverage statistics\n')
        f.write('### Schemaless operators\n')
        f.write('<details>\n')
        f.write(
            '<summary>There are {} schemaless operators</summary>\n\n'.format(
                num_schemaless))
        for n in sorted(schemaless):
            f.write('* ' + n + '\n')
        f.write('</details>\n\n')


def gen_coverage_sets(source_dir):
    covered_ops = gen_covered_ops(source_dir)

    not_covered_ops = set()
    schemaless_ops = []
    for op_name in core._GetRegisteredOperators():
        s = OpSchema.get(op_name)

        if s is not None and s.private:
            continue
        if s:
            if op_name not in covered_ops:
                not_covered_ops.add(op_name)
        else:
            if op_name.find("_ENGINE_") == -1:
                schemaless_ops.append(op_name)
    return (covered_ops, not_covered_ops, schemaless_ops)


def gen_covered_ops(source_dir):
    def parse_proto(x):
        proto = caffe2_pb2.OperatorDef()
        proto.ParseFromString(x)
        return proto

    covered = set()
    for f in os.listdir(source_dir):
        zipfile = os.path.join(source_dir, f)
        if not os.path.isfile(zipfile):
            continue
        temp_dir = tempfile.mkdtemp()
        with ZipFile(zipfile) as z:
            z.extractall(temp_dir)
        op_path = os.path.join(temp_dir, 'op.pb')
        with open(op_path, 'rb') as f:
            loaded_op = f.read()
        op_proto = parse_proto(loaded_op)
        covered.add(op_proto.type)

        index = 0
        grad_path = os.path.join(temp_dir, 'grad_{}.pb'.format(index))
        while os.path.isfile(grad_path):
            with open(grad_path, 'rb') as f:
                loaded_grad = f.read()
            grad_proto = parse_proto(loaded_grad)
            covered.add(grad_proto.type)
            index += 1
            grad_path = os.path.join(temp_dir, 'grad_{}.pb'.format(index))
    return covered
