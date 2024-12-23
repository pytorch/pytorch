from setuptools import Command
from setuptools.warnings import SetuptoolsDeprecationWarning


def __getattr__(name):
    if name == 'test':
        SetuptoolsDeprecationWarning.emit(
            "The test command is disabled and references to it are deprecated.",
            "Please remove any references to `setuptools.command.test` in all "
            "supported versions of the affected package.",
            due_date=(2024, 11, 15),
            stacklevel=2,
        )
        return _test
    raise AttributeError(name)


class _test(Command):
    """
    Stub to warn when test command is referenced or used.
    """

    description = "stub for old test command (do not use)"

    user_options = [
        ('test-module=', 'm', "Run 'test_suite' in specified module"),
        (
            'test-suite=',
            's',
            "Run single test, case or suite (e.g. 'module.test_suite')",
        ),
        ('test-runner=', 'r', "Test runner to use"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        raise RuntimeError("Support for the test command was removed in Setuptools 72")
