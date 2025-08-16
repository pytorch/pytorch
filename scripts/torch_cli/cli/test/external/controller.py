from cement import Controller, ex
from cli.test.external.vllm_test import VllmTestRunner


class ExternalTestController(Controller):
    class Meta:
        label = "test_external"
        stacked_on = "test"
        stacked_type = "nested"
        help = "tests external libraries."
        aliases = ["external"]

    @ex(
        help="test vllm",
        arguments=[
            (
                ["--test-name"],
                {
                    "help": "test name to run tests, for example, 'vllm_basic_correctness_test'",
                    "dest": "test_name",
                    "default": "",
                    "type": str,
                    "required": True,
                },
            )
        ],
    )
    def vllm(self):
        pargs = self.app.pargs
        test_name = pargs.test_name
        runner = VllmTestRunner()
        runner.run([test_name])
