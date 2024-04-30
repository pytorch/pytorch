import os
import pytest
import _pytest
import _pytest.config
import re

known_tests = dict()
discovered_tests = set()

skipped = set()
passed = set()
failed = set()

def main():
    global known_tests
    # known_tests['TestMkldnnCPU.test_mul_cpu'] = 'test/dynamo_expected_failures/TestMkldnnCPU.test_mul_cpu'
    ## for f in os.listdir('test/dynamo_skips'):
    ##     known_tests[f] = os.path.join('test/dynamo_skips', f)
    ## for f in os.listdir('test/dynamo_expected_failures'):
    ##     known_tests[f] = os.path.join('test/dynamo_expected_failures', f)

    ## with open('xxx.txt', 'r') as f:
    ##     import json
    ##     known_tests = json.load(f)
    ## known_tests = {
    ##     k: v
    ##     for i, (k, v) in enumerate(known_tests.items())
    ##     if False
    ##     or 245 <= i < 247
    ## }

    ## known_tests['TestSparseCPU.test_log1p_cpu_uint8'] = 'test/dynamo_skips/TestSparseCPU.test_log1p_cpu_uint8'
    ## with open('xxx.txt', 'w') as f:
    ##     import json
    ##     json.dump(known_tests, f)

    known_tests["DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_free_tensor_dynamic_shapes"] = "test/dynamo_skips/DynamicShapesFuncTorchHigherOrderOpTests.test_vmap_free_tensor_dynamic_shapes"
    known_tests["TestSparseCPU.test_log1p_cpu_uint8"] = "test/dynamo_skips/TestSparseCPU.test_log1p_cpu_uint8"

    print(f"known tests: {len(known_tests)}")
    # return

    classes = set()
    for f in known_tests.keys():
        c = f.split('.')[0]
        if c.endswith('CPU'):
            classes.add(c[:-3])
        elif c.endswith('CUDA'):
            classes.add(c[:-4])
        elif c.startswith('DynamicShapes'):
            classes.add("DynamicShapes")
        elif c.startswith('NonStrictExport'):
            classes.add("NonStrictExport")
        elif c.startswith('RetraceExport'):
            classes.add("RetraceExport")
        elif c.startswith('SerDesExport'):
            classes.add("SerDesExport")
        elif c == 'OptimizerTests':
            classes.add("Optimizer")
        else:
            classes.add(c)
    print(f"{len(classes)} classes")

    total = 0
    found = set()
    testfiles = set()
    for path, names, files in os.walk('test'):
        for f in files:
            if not f.endswith('.py'):
                continue
            f = os.path.join(path, f)
            total += 1

            with open(f, 'r') as h:
                data = h.read()

            for m in re.findall(r"^class (\w+)", data, re.M):
                if m in classes:
                    found.add(m)
                    testfiles.add(f)

    if "DynamicShapes" in classes:
        found.add("DynamicShapes")
        testfiles.add("test/inductor/test_torchinductor_dynamic_shapes.py")
        testfiles.add("test/dynamo/test_dynamic_shapes.py")
    if "NonStrictExport" in classes:
        found.add("NonStrictExport")
        testfiles.add("test/export/test_export_nonstrict.py")
    if "Optimizer" in classes:
        found.add("Optimizer")
        testfiles.add("test/test_optim.py")
    if "RetraceExport" in classes:
        found.add("RetraceExport")
        testfiles.add("test/export/test_retraceability.py")
    if "SerDesExport" in classes:
        found.add("SerDesExport")
        testfiles.add("test/export/test_serdes.py")

    print(f"{total} files")
    print(f"{len(testfiles)} test files")
    print(f"{len(found)} classes found")
    print(f"missing: {len(classes - found)}\n", sorted(classes - found))

    print("testfiles", testfiles)

    os.environ['PYTORCH_TEST_WITH_DYNAMO'] = '1'
    args = ['--no-summary', '--continue-on-collection-errors'] + sorted(testfiles)
    config = _pytest.config.get_config(args)
    pluginmanager = config.pluginmanager
    pluginmanager.register(MyPlugin())
    config = pluginmanager.hook.pytest_cmdline_parse(pluginmanager=pluginmanager, args=args)
    config.hook.pytest_cmdline_main(config=config)

    print("SKIPPED", len(skipped))
    print("FAILED", len(failed))
    undiscovered = set(known_tests.keys()) - discovered_tests
    if len(undiscovered) < 100:
        print("UNDISCOVERED", sorted(undiscovered))
    else:
        print("UNDISCOVERED", len(undiscovered))

    print("PASSED", len(passed))
    for p in sorted(passed):
        print(f"  {p}")

class MyPlugin:
    def pytest_collection_modifyitems(self, items, config):
        selected = []
        deselected = []
        for item in items:
            if not item or not item.cls:
                continue
            name = f"{item.cls.__name__}.{item.name}"
            if name in known_tests:
                selected.append(item)
                discovered_tests.add(name)
                try:
                    os.unlink(known_tests[name])
                except FileNotFoundError:
                    pass
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)

    def pytest_report_teststatus(self, report, config):
        # print(f"*** pytest_report_teststatus({report.when}, {report.outcome})")

        def revert(name):
            with open(known_tests[name], "w") as f:
                pass

        _, cls, name = report.nodeid.split('::')
        name = f"{cls}.{name}"
        assert name in known_tests, f"name: {name}"

        match (report.when, report.outcome):
            case (('setup', 'skipped') |
                  ('setup', 'failed') |
                  ('call', 'failed') |
                  ('call', 'skipped')):
                match report.outcome:
                    case 'skipped':
                        skipped.add(name)
                    case 'failed':
                        failed.add(name)
                revert(name)

            case (('teardown', 'passed') |
                  ('setup', 'passed')):
                pass

            case ('call', 'passed'):
                passed.add(name)

            case _:
                assert False, f"REPORT: ({report.when}, {report.outcome})"


main()
