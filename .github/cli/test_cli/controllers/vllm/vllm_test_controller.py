from cement import Controller, ex
from test_cli.controllers.vllm.test_runner import VllmTestRunner
class VllmTestController(Controller):
    class Meta:
        label = "vllm"
        stacked_on = "base"
        stacked_type = "nested"
        help = "Build external libraries. to run the build, do cli.py external --target vllm run, to see the details of help: \
        cli.py external --target vllm help "
        arguments = [
            (
                ["--test-name"],
                {
                    "help": "test name to run tests",
                    "dest": "test_name",
                    "default":"./results",
                    "type": str,
                    "required": False,
                },
            )
        ]

    @ex(help="test vllm")
    def run(self):
        pargs = self.app.pargs
        test_name = pargs.test_name
        runner = VllmTestRunner()
        runner.run([test_name])
