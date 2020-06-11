import cimodel.data.simple.util.branch_filters


class AndroidJob:
    def __init__(self, job_name, template_name, dependencies):
        self.job_name = job_name
        self.template_name = template_name
        self.dependencies = dependencies

    def gen_tree(self):

        props_dict = {
            "filters": cimodel.data.simple.util.branch_filters.gen_filter_dict(),
            "name": self.job_name,
            "requires": self.dependencies,
        }

        return [{self.template_name: props_dict}]


WORKFLOW_DATA = [
    AndroidJob(
        "pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-build-x86_32",
        "pytorch_android_gradle_build-x86_32",
        ["pytorch_linux_xenial_py3_clang5_android_ndk_r19c_x86_32_build"]),
    AndroidJob(
        "pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-build",
        "pytorch_android_gradle_build",
        ["pytorch_linux_xenial_py3_clang5_android_ndk_r19c_x86_32_build",
         "pytorch_linux_xenial_py3_clang5_android_ndk_r19c_x86_64_build",
         "pytorch_linux_xenial_py3_clang5_android_ndk_r19c_arm_v7a_build",
         "pytorch_linux_xenial_py3_clang5_android_ndk_r19c_arm_v8a_build"]),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
