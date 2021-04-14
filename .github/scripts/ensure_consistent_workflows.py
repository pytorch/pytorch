#!/usr/bin/env python
'''
This file verifies that the workflows that are potentially canceled in our cancel_redundant_workflow.yml
match the workflows we have running on pull requests (found in .github/workflows). This way, anytime a
workflow is added or removed, people can be reminded to modify the cancel_redundant_workflow.yml accordingly.
'''


import yaml
from pathlib import Path


workflow_paths = (Path(__file__).parent.parent / 'workflows').rglob('*')
workflows = []
for path in workflow_paths:
    if path.suffix in {'.yml', '.yaml'}:
        with open(path) as f:
            data = yaml.safe_load(f)
            assert 'name' in data, 'Every GHA workflow must have a name.'
            # The reason the below key is True is because yaml parses 'on:' as a boolean.
            # Since the solutions to maintain the key as 'on' seems more complicated than it's worth,
            # I decided to stick with having True as a key instead of hacking my way through YAML
            # implementation that is bound to change and become better anyway. More context below:
            # https://stackoverflow.com/questions/36463531/pyyaml-automatically-converting-certain-keys-to-boolean-values
            if 'pull_request' in data[True]:
                workflows.append(data['name'])


with open('.github/workflows/cancel_redundant_workflows.yml') as f:
    data = yaml.safe_load(f)
    workflows_to_cancel = data[True]['workflow_run']['workflows']
    missing = list(set(workflows) - set(workflows_to_cancel))
    should_remove = list(set(workflows_to_cancel) - set(workflows))
    if missing == [] and should_remove == []:
        exit()
    if missing:
        print(f'ACTION: Please add these to the cancel_redundant_workflow list: {missing}')
    if should_remove:
        print(f'ACTION: Please remove these from the cancel_redundant_workflow list: {should_remove}')
    raise RuntimeError('Please follow the above instructions to make our GHA workflows consistent')
