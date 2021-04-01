#!/usr/bin/env python

import os
import yaml

rootdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'workflows')
for subdir, _, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        if filepath.endswith(('.yml', '.yaml')):
            with open(filepath) as f:
                data = yaml.safe_load(f)
                assert 'name' in data, 'Every GHA workflow must have a name.'
                # The reason the below key is True is because yaml parses 'on:' as a boolean.
                # Since the solutions to maintain the key as 'on' seems more complicated than it's worth,
                # I decided to stick with having True as a key instead of hacking my way through YAML
                # implementation that is bound to change and become better anyway. More context below:
                # https://stackoverflow.com/questions/36463531/pyyaml-automatically-converting-certain-keys-to-boolean-values
                if 'pull_request' in data[True]:
                    print(data['name'])
