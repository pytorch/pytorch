from cimodel.data.simple.util.versions import MultiPartVersion
import cimodel.lib.miniutils as miniutils

XCODE_VERSION = MultiPartVersion([12, 0, 0])


class ArchVariant:
    def __init__(self, name, custom_build_name=""):
        self.name = name
        self.custom_build_name = custom_build_name

    def render(self):
        extra_parts = [self.custom_build_name] if len(self.custom_build_name) > 0 else []
        return "_".join([self.name] + extra_parts)


def get_platform(arch_variant_name):
    return "SIMULATOR" if arch_variant_name == "x86_64" else "OS"


class IOSJob:
    def __init__(self, xcode_version, arch_variant, is_org_member_context=True, extra_props=None):
        self.xcode_version = xcode_version
        self.arch_variant = arch_variant
        self.is_org_member_context = is_org_member_context
        self.extra_props = extra_props

    def gen_name_parts(self, with_version_dots):

        version_parts = self.xcode_version.render_dots_or_parts(with_version_dots)
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
    IOSJob(XCODE_VERSION, ArchVariant("x86_64"), is_org_member_context=False),
    IOSJob(XCODE_VERSION, ArchVariant("x86_64", "lite_interpreter"), is_org_member_context=False, extra_props={
        "lite_interpreter": miniutils.quote(str(int(True)))}),
    IOSJob(XCODE_VERSION, ArchVariant("arm64")),
    IOSJob(XCODE_VERSION, ArchVariant("arm64", "metal"), extra_props={"use_metal": miniutils.quote(str(int(True)))}),
    IOSJob(XCODE_VERSION, ArchVariant("arm64", "lite_interpreter"), extra_props={
        "lite_interpreter": miniutils.quote(str(int(True)))}),
    IOSJob(XCODE_VERSION, ArchVariant("arm64", "custom"), extra_props={"op_list": "mobilenetv2.yaml"}),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
