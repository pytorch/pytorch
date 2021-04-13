#!/usr/bin/env python

import yaml
from pathlib import Path

workflow_paths = (Path(__file__).parent.parent / 'workflows').rglob('*')
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
                print(data['name'])
