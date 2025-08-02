from cement import Controller, ex
from test_cli.external.vllm_test import VllmTestRunner
class ExternalTestController(Controller):
    class Meta:
        label = "external"
        stacked_on = "base"
        stacked_type = "nested"
        help = "tests vllm external libraries. notice this must be run from the root of the pytorch repo"
        arguments = [
            (
                ["--test-name"],
                {
                    "help": "test name to run tests, for example, 'vllm_basic_correctness_test'",
                    "dest": "test_name",
                    "default":"",
                    "type": str,
                    "required": True,
                },
            )
        ]

    @ex(help="test vllm")
    def vllm(self):
        pargs = self.app.pargs
        test_name = pargs.test_name
        runner = VllmTestRunner()
        runner.run([test_name])
