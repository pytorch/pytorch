import cimodel.lib.miniutils as miniutils
from cimodel.data.simple.util.versions import MultiPartVersion, CudaVersion
from cimodel.data.simple.util.docker_constants import DOCKER_IMAGE_BASIC, DOCKER_IMAGE_CUDA_10_2


class GeConfigTestJob:
    def __init__(self,
                 py_version,
                 gcc_version,
                 cuda_version,
                 variant_parts,
                 extra_requires,
                 use_cuda_docker=False,
                 build_env_override=None):

        self.py_version = py_version
        self.gcc_version = gcc_version
        self.cuda_version = cuda_version
        self.variant_parts = variant_parts
        self.extra_requires = extra_requires
        self.use_cuda_docker = use_cuda_docker
        self.build_env_override = build_env_override

    def get_all_parts(self, with_dots):

        maybe_py_version = self.py_version.render_dots_or_parts(with_dots) if self.py_version else []
        maybe_gcc_version = self.gcc_version.render_dots_or_parts(with_dots) if self.gcc_version else []
        maybe_cuda_version = self.cuda_version.render_dots_or_parts(with_dots) if self.cuda_version else []

        common_parts = [
            "pytorch",
            "linux",
            "xenial",
        ] + maybe_cuda_version + maybe_py_version + maybe_gcc_version

        return common_parts + self.variant_parts

    def gen_tree(self):

        resource_class = "gpu.medium" if self.use_cuda_docker else "large"
        docker_image = DOCKER_IMAGE_CUDA_10_2 if self.use_cuda_docker else DOCKER_IMAGE_BASIC
        full_name = "_".join(self.get_all_parts(False))
        build_env = self.build_env_override or "-".join(self.get_all_parts(True))

        props_dict = {
            "name": full_name,
            "build_environment": build_env,
            "requires": self.extra_requires,
            "resource_class": resource_class,
            "docker_image": docker_image,
        }

        if self.use_cuda_docker:
            props_dict["use_cuda_docker_runtime"] = miniutils.quote(str(1))

        return [{"pytorch_linux_test": props_dict}]


WORKFLOW_DATA = [
    GeConfigTestJob(
        MultiPartVersion([3, 6], "py"),
        MultiPartVersion([5, 4], "gcc"),
        None,
        ["jit_legacy", "test"],
        ["pytorch_linux_xenial_py3_6_gcc5_4_build"]),
    GeConfigTestJob(
        None,
        None,
        CudaVersion(10, 2),
        ["cudnn7", "py3", "jit_legacy", "test"],
        ["pytorch_linux_xenial_cuda10_2_cudnn7_py3_gcc7_build"],
        use_cuda_docker=True,
    ),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
