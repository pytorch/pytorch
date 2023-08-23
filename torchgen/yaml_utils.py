# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[misc]

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper  # type: ignore[misc]
YamlDumper = Dumper


# A custom loader for YAML that errors on duplicate keys.
# This doesn't happen by default: see https://github.com/yaml/pyyaml/issues/165
class YamlLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            assert (
                key not in mapping
            ), f"Found a duplicate key in the yaml. key={key}, line={node.start_mark.line}"
            mapping.append(key)
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        return mapping
