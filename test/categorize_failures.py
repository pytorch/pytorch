from argparse import ArgumentParser
from collections import defaultdict
from json import dumps
from xml.etree import ElementTree
from re import sub
from sys import stderr
from os.path import join, exists

parser = ArgumentParser()
parser.add_argument(
    "-s",
    "--short",
    action='store_true',
    default=False,
)
args = parser.parse_args()


def xml2dict(node):
    dict = {}
    # Convert attributes
    for attribute in node.attrib.keys():
        dict[attribute] = node.get(attribute)

    # Convert text
    if node.text and node.text != '' and node.text != ' ' and node.text != '\n':
        if 'text' in dict:
            print('Warning: node already contains "text" attribute', file=stderr)
        dict['text'] = node.text

    # Convert children
    for child in node:
        if not child.tag in dict:
            dict[child.tag] = []
        elif not isinstance(dict[child.tag], list):
            print(f'Warning: node already contains "{child.tag}" attribute', file=stderr)
        dict[child.tag].append(xml2dict(child))

    return dict


def filter_message(message):
    regex1 = r"(\w:\\.*:\d+: )"
    regex2 = r"(\s*\(.*\).*)"
    return sub(f"{regex1}|{regex2}", "", message)


def get_xfailed_cause(testcase):
    message = testcase['skipped'][0]['message']
    if message:
        return message
    system_err = testcase.get('system-err', [])
    if len(system_err) > 0:
        return filter_message(system_err[0]['text'].split('\n')[0])
    system_out = testcase.get('system-out', [])
    if len(system_out):
        return filter_message(system_out[0]['text'].split('\n')[0])
    return ''


def get_skipped_cause(testcase):
    return testcase['skipped'][0]['message'].split('\n')[0]


def get_failed_cause(testcase):
    return testcase['failure'][0]['message'].split('\n')[0]


testsuites = [
    'autograd',
    'modules',
    'nn',
    'ops',
    'ops_fwd_gradients',
    'ops_gradients',
    'ops_jit',
    'torch'
]

testcases = []
for testsuite in testsuites:
    filename = join('results', f'test_{testsuite}.xml')
    if not exists(filename):
        print(f'Test suite results do not exist: {filename}', file=stderr)
    root = xml2dict(ElementTree.parse(filename).getroot())
    for tests in root['testsuite']:
        testcases.extend(tests['testcase'])


# Categorize passed tests
passed = list(filter(lambda x: 'skipped' not in x and 'failure' not in x, testcases))

print(f'Passed: {len(passed)}', file=stderr)

# Categorize xfailed tests
xfailed = list(filter(lambda x: 'skipped' in x and 'pytest.xfail' == x['skipped'][0]['type'], testcases))
xfailed_causes = defaultdict(list)
for testcase in xfailed:
    xfailed_causes[get_xfailed_cause(testcase)].append(f"{testcase['classname']}::{testcase['name']}")
xfailed_causes = {k: {'count': len(v), 'tests': v} for k, v in xfailed_causes.items()}
xfailed_causes = dict(sorted(xfailed_causes.items(), key=lambda x: x[1]['count'], reverse=True))
if (args.short):
    xfailed_causes = {k: v['count'] for k, v in xfailed_causes.items()}

print(f'Xfailed: {len(xfailed)}', file=stderr)

# Categorize skipped tests
skipped = list(filter(lambda x: 'skipped' in x and 'pytest.skip' == x['skipped'][0]['type'], testcases))
skipped_causes = defaultdict(list)
for testcase in skipped:
    skipped_causes[get_skipped_cause(testcase)].append(f"{testcase['classname']}::{testcase['name']}")
skipped_causes = {k: {'count': len(v), 'tests': v} for k, v in skipped_causes.items()}
skipped_causes = dict(sorted(skipped_causes.items(), key=lambda x: x[1]['count'], reverse=True))
if (args.short):
    skipped_causes = {k: v['count'] for k, v in skipped_causes.items()}

print(f'Skipped: {len(skipped)}', file=stderr)

# Categorize failed tests
failed = list(filter(lambda x: 'failure' in x, testcases))
failed_causes = defaultdict(list)
for testcase in failed:
    failed_causes[get_failed_cause(testcase)].append(f"{testcase['classname']}::{testcase['name']}")
failed_causes = {k: {'count': len(v), 'tests': v} for k, v in failed_causes.items()}
failed_causes = dict(sorted(failed_causes.items(), key=lambda x: x[1]['count'], reverse=True))
if (args.short):
    failed_causes = {k: v['count'] for k, v in failed_causes.items()}

print(f'Failed: {len(failed)}', file=stderr)

results = {
    'xfailed': xfailed_causes,
    'skipped': skipped_causes,
    'failed': failed_causes
}

print(dumps(results, indent=2))
