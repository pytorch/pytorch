from cement import Controller, ex
from .external_vllm_build import build_vllm


class ExternalBuildController(Controller):
    class Meta:
        label = "external"
        stacked_on = "base"
        stacked_type = "nested"
        help = "Build external libraries. to run the build, do cli.py external --target vllm run, to see the details of help: \
        cli.py external --target vllm help "
        arguments = [
            (
                ["--target"],
                {
                    "help": "Component to build (e.g. vllm)",
                    "dest": "target",
                    "choices": ["vllm"],
                },

            ),
            (
                ["--artifact-dir"],
                {
                    "help": "Path to a local folder where artifacts from external builds will be stored",
                    "dest": "artifact_dir",
                    "default":"./results",
                    "type": str,
                    "required": False,
                },
            )
        ]

    @ex(help="Build external component")
    def run(self):
        target = self.app.pargs.target
        print(f"[INFO] Target: {target}")
        artifact_dir = self.app.pargs.artifact_dir

        print(f"[INFO] Plan to save artifacts to: {artifact_dir}")
        match target:
            case "vllm":
                build_vllm()
            case _:
                print(f"[ERROR] Unknown target: {target}")

    @ex(help="Show detailed help for a build target")
    def help(self):
        target = self.app.pargs.target
        match target:
            case "vllm":
                print(
                    """
                    [HELP] Extended help for target `vllm`

                    This target builds the vLLM wheel using several environment variables:

                      TAG                     Image tag name                 [default: vllm-wheels]
                      CUDA_VERSION            CUDA version                   [default: 12.8.0]
                      PYTHON_VERSION          Python version                 [default: 3.12]
                      MAX_JOBS                Max parallel jobs              [default: 32]
                      TARGET                  Docker build target            [default: export-wheels]
                      SCCACHE_BUCKET_NAME     sccache bucket name            [default: ""]
                      SCCACHE_REGION_NAME     sccache region name            [default: ""]
                      TORCH_CUDA_ARCH_LIST    Torch CUDA architectures       [default: 8.6;8.9]

                    Example:

                        TAG=nightly-vllm CUDA_VERSION=12.8.0 cli.py external --target vllm
                """
                )
            case _:
                print(f"[ERROR] No extended help available for target: {target}")
