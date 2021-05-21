#!/usr/bin/env python3
'''
This file verifies that the workflows that are potentially canceled in our cancel_redundant_workflow.yml
match the workflows we have running on pull requests (found in .github/workflows). This way, anytime a
workflow is added or removed, people can be reminded to modify the cancel_redundant_workflow.yml accordingly.
'''


import ruamel.yaml
from pathlib import Path


yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']
yaml.default_flow_style = False


if __name__ == '__main__':
    workflow_paths = (Path(__file__).parent.parent / 'workflows').rglob('*')
    workflows = []
    for path in workflow_paths:
        if path.suffix in {'.yml', '.yaml'}:
            with open(path) as f:
                data = yaml.load(f)
                assert 'name' in data, 'Every GHA workflow must have a name.'
                if 'pull_request' in data['on']:
                    workflows.append(data['name'])

    with open('.github/workflows/cancel_redundant_workflows.yml', 'r') as f:
        data = yaml.load(f)

    # Replace workflows to cancel
    data['on']['workflow_run']['workflows'] = sorted(workflows)

    with open('.github/workflows/cancel_redundant_workflows.yml', 'w') as f:
        yaml.dump(data, f)
