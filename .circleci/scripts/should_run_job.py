import argparse
import re
import sys

# Modify this variable if you want to change the set of default jobs
# which are run on all pull requests.
#
# WARNING: Actually, this is a lie; we're currently also controlling
# the set of jobs to run via the Workflows filters in CircleCI config.

default_set = set([
    'pytorch-windows-ws2019-cuda10.1-cudnn7-py3',

])

# Collection of jobs that are *temporarily* excluded from running on PRs.
# Use this if there is a long-running job breakage that we can't fix with a
# single revert.
skip_override = {
    # example entry:
    # 'pytorch-cpp-doc-push': "https://github.com/pytorch/pytorch/issues/<related issue>"
}

# Takes in commit message to analyze via stdin
#
# This script will query Git and attempt to determine if we should
# run the current CI job under question
#
# NB: Try to avoid hard-coding names here, so there's less place to update when jobs
# are updated/renamed
#
# Semantics in the presence of multiple tags:
#   - Let D be the set of default builds
#   - Let S be the set of explicitly specified builds
#   - Let O be the set of temporarily skipped builds
#   - Run S \/ (D - O)

parser = argparse.ArgumentParser()
parser.add_argument('build_environment')
args = parser.parse_args()

commit_msg = sys.stdin.read()

# Matches anything that looks like [foo ci] or [ci foo] or [foo test]
# or [test foo]
RE_MARKER = re.compile(r'\[(?:([^ \[\]]+) )?(?:ci|test)(?: ([^ \[\]]+))?\]')

markers = RE_MARKER.finditer(commit_msg)

for m in markers:
    if m.group(1) and m.group(2):
        print("Unrecognized marker: {}".format(m.group(0)))
        continue
    spec = m.group(1) or m.group(2)
    if spec is None:
        print("Unrecognized marker: {}".format(m.group(0)))
        continue
    if spec in args.build_environment or spec == 'all':
        print("Accepting {} due to commit marker {}".format(args.build_environment, m.group(0)))
        sys.exit(0)

skip_override_set = set(skip_override.keys())
should_run_set = default_set - skip_override_set
for spec in should_run_set:
    if spec in args.build_environment:
        print("Accepting {} as part of default set".format(args.build_environment))
        sys.exit(0)

print("Rejecting {}".format(args.build_environment))
for spec, issue in skip_override.items():
    if spec in args.build_environment:
        print("This job is temporarily excluded from running on PRs. Reason: {}".format(issue))
        break
sys.exit(1)
