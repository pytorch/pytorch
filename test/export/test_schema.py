# Owner(s): ["oncall: export"]
from torch._export.serde.schema_check import (
    _Commit,
    _diff_schema,
    check,
    SchemaUpdateError,
    update_schema,
)
from torch.testing._internal.common_utils import IS_FBCODE, run_tests, TestCase


class TestSchema(TestCase):
    def test_schema_compatibility(self):
        msg = """
Detected an invalidated change to export schema. Please run the following script to update the schema:
Example(s):
    python scripts/export/update_schema.py --prefix <path_to_torch_development_diretory>
        """

        if IS_FBCODE:
            msg += """or
    buck run caffe2:export_update_schema -- --prefix /data/users/$USER/fbsource/fbcode/caffe2/
            """
        try:
            commit = update_schema()
        except SchemaUpdateError as e:
            self.fail(f"Failed to update schema: {e}\n{msg}")

        self.assertEqual(commit.checksum_base, commit.checksum_result, msg)

    def test_schema_diff(self):
        additions, subtractions = _diff_schema(
            {
                "Type0": {"kind": "struct", "fields": {}},
                "Type2": {
                    "kind": "struct",
                    "fields": {
                        "field0": {"type": ""},
                        "field2": {"type": ""},
                        "field3": {"type": "", "default": "[]"},
                    },
                },
            },
            {
                "Type2": {
                    "kind": "struct",
                    "fields": {
                        "field1": {"type": "", "default": "0"},
                        "field2": {"type": "", "default": "[]"},
                        "field3": {"type": ""},
                    },
                },
                "Type1": {"kind": "struct", "fields": {}},
            },
        )

        self.assertEqual(
            additions,
            {
                "Type1": {"kind": "struct", "fields": {}},
                "Type2": {
                    "fields": {
                        "field1": {"type": "", "default": "0"},
                        "field2": {"default": "[]"},
                    },
                },
            },
        )
        self.assertEqual(
            subtractions,
            {
                "Type0": {"kind": "struct", "fields": {}},
                "Type2": {
                    "fields": {
                        "field0": {"type": ""},
                        "field3": {"default": "[]"},
                    },
                },
            },
        )

    def test_schema_check(self):
        # Adding field without default value
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                    "field1": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_result="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_base="",
            cpp_header="",
            cpp_header_path="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [4, 1])

        # Removing field
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {},
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_result="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_base="",
            cpp_header="",
            cpp_header_path="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [4, 1])

        # Adding field with default value
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                    "field1": {"type": "", "default": "[]"},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_result="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_base="",
            cpp_header="",
            cpp_header_path="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [3, 3])

        # Changing field type
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": "int"},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }

        with self.assertRaises(SchemaUpdateError):
            _diff_schema(dst, src)

        # Adding new type.
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "Type1": {"kind": "struct", "fields": {}},
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_result="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_base="",
            cpp_header="",
            cpp_header_path="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [3, 3])

        # Removing a type.
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_result="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_base="",
            cpp_header="",
            cpp_header_path="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [3, 3])

        # Adding new field in union.
        dst = {
            "Type2": {
                "kind": "union",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "union",
                "fields": {
                    "field0": {"type": ""},
                    "field1": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_result="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_base="",
            cpp_header="",
            cpp_header_path="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [3, 3])

        # Removing a field in union.
        dst = {
            "Type2": {
                "kind": "union",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "union",
                "fields": {},
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_result="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_base="",
            cpp_header="",
            cpp_header_path="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [4, 1])


if __name__ == "__main__":
    run_tests()
