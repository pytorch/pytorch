from unittest import main, TestCase

from coerce_tool_input import coerced


class TestCoerceToolInput(TestCase):
    def test_string_issue_number_becomes_int(self):
        out = coerced({"owner": "pytorch", "issue_number": "195701"})
        self.assertEqual(out, {"owner": "pytorch", "issue_number": 195701})

    def test_hash_prefix_is_stripped(self):
        self.assertEqual(coerced({"issue_number": "#195701"})["issue_number"], 195701)

    def test_json_string_labels_become_list(self):
        # Seen in ciforge run 34505448200: the whole list arrived as one string.
        out = coerced(
            {"issue_number": 596, "labels": '["module: cuda", "triage review"]'}
        )
        self.assertEqual(out["labels"], ["module: cuda", "triage review"])

    def test_well_typed_input_is_untouched(self):
        self.assertIsNone(coerced({"issue_number": 195701, "labels": ["module: cuda"]}))
        self.assertIsNone(coerced({"query": "is:issue"}))

    def test_garbage_is_left_for_the_server_to_reject(self):
        self.assertIsNone(coerced({"issue_number": "abc", "labels": "module: cuda"}))
        self.assertIsNone(coerced({"labels": '{"a": 1}'}))


if __name__ == "__main__":
    main()
