#!/usr/bin/env python3

import os
import re


def main() -> None:
    ref = os.environ['GITHUB_REF']
    m = re.match(r'^refs/(\w+)/(.*)$', ref)
    if m:
        category, stripped = m.groups()
        if category == 'heads':
            print(f'::set-output name=branch::{stripped}')
        elif category == 'pull':
            print(f'::set-output name=branch::pull/{stripped.split("/")[0]}')
        elif category == 'tags':
            print(f'::set-output name=tag::{stripped}')


if __name__ == '__main__':
    main()
