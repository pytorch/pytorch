import unittest

from tools import extract_scripts

requirements_sh = '''
#!/usr/bin/env bash
set -eo pipefail
pip install -r requirements.txt
'''.strip()

hello_sh = '''
#!/usr/bin/env sh
set -e
echo hello world
'''.strip()


class TestExtractScripts(unittest.TestCase):
    def test_extract_none(self) -> None:
        self.assertEqual(
            extract_scripts.extract({
                'name': 'Checkout PyTorch',
                'uses': 'actions/checkout@v2',
            }),
            None,
        )

    def test_extract_run_default_bash(self) -> None:
        self.assertEqual(
            extract_scripts.extract({
                'name': 'Install requirements',
                'run': 'pip install -r requirements.txt',
            }),
            {
                'extension': '.sh',
                'script': requirements_sh,
            },
        )

    def test_extract_run_sh(self) -> None:
        self.assertEqual(
            extract_scripts.extract({
                'name': 'Hello world',
                'run': 'echo hello world',
                'shell': 'sh',
            }),
            {
                'extension': '.sh',
                'script': hello_sh,
            },
        )

    def test_extract_run_py(self) -> None:
        self.assertEqual(
            extract_scripts.extract({
                'name': 'Hello world',
                'run': 'print("Hello!")',
                'shell': 'python',
            }),
            {
                'extension': '.py',
                'script': 'print("Hello!")',
            },
        )

    def test_extract_github_script(self) -> None:
        self.assertEqual(
            # https://github.com/actions/github-script/tree/v3.1.1#reading-step-results
            extract_scripts.extract({
                'uses': 'actions/github-script@v3',
                'id': 'set-result',
                'with': {
                    'script': 'return "Hello!"',
                    'result-encoding': 'string',
                },
            }),
            {
                'extension': '.js',
                'script': 'return "Hello!"',
            },
        )


if __name__ == '__main__':
    unittest.main()
