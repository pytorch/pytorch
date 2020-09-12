from cimodel.data.simple.util.versions import MultiPartVersion


IOS_VERSION = MultiPartVersion([11, 2, 1])


class ArchVariant:
    def __init__(self, name, is_custom=False):
        self.name = name
        self.is_custom = is_custom

    def render(self):
        extra_parts = ["custom"] if self.is_custom else []
        return "_".join([self.name] + extra_parts)


def get_platform(arch_variant_name):
    return "SIMULATOR" if arch_variant_name == "x86_64" else "OS"


class IOSJob:
    def __init__(self, ios_version, arch_variant, is_org_member_context=True, extra_props=None):
        self.ios_version = ios_version
        self.arch_variant = arch_variant
        self.is_org_member_context = is_org_member_context
        self.extra_props = extra_props

    def gen_name_parts(self, with_version_dots):

        version_parts = self.ios_version.render_dots_or_parts(with_version_dots)
        build_variant_suffix = "_".join([self.arch_variant.render(), "build"])

        return [
            "pytorch",
            "ios",
        ] + version_parts + [
            build_variant_suffix,
        ]

    def gen_job_name(self):
        return "_".join(self.gen_name_parts(False))

    def gen_tree(self):

        platform_name = get_platform(self.arch_variant.name)

        props_dict = {
            "build_environment": "-".join(self.gen_name_parts(True)),
            "ios_arch": self.arch_variant.name,
            "ios_platform": platform_name,
            "name": self.gen_job_name(),
        }

        if self.is_org_member_context:
            props_dict["context"] = "org-member"

        if self.extra_props:
            props_dict.update(self.extra_props)

        return [{"pytorch_ios_build": props_dict}]


WORKFLOW_DATA = [
    IOSJob(IOS_VERSION, ArchVariant("x86_64"), is_org_member_context=False),
    # IOSJob(IOS_VERSION, ArchVariant("arm64")),
    # IOSJob(IOS_VERSION, ArchVariant("arm64", True), extra_props={"op_list": "mobilenetv2.yaml"}),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
