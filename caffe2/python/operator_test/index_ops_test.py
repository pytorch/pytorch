



from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np
import tempfile


class TestIndexOps(TestCase):
    def _test_index_ops(self, entries, dtype, index_create_op):
        workspace.RunOperatorOnce(core.CreateOperator(
            index_create_op,
            [],
            ['index'],
            max_elements=10))
        my_entries = np.array(
            [entries[0], entries[1], entries[2]], dtype=dtype)

        workspace.FeedBlob('entries', my_entries)
        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexLoad',
            ['index', 'entries'],
            ['index']))
        query1 = np.array(
            [entries[0], entries[3], entries[0], entries[4]],
            dtype=dtype)

        workspace.FeedBlob('query1', query1)
        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexGet',
            ['index', 'query1'],
            ['result1']))
        result1 = workspace.FetchBlob('result1')
        np.testing.assert_array_equal([1, 4, 1, 5], result1)

        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexFreeze',
            ['index'],
            ['index']))

        query2 = np.array(
            [entries[5], entries[4], entries[0], entries[6], entries[7]],
            dtype=dtype)
        workspace.FeedBlob('query2', query2)
        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexGet',
            ['index', 'query2'],
            ['result2']))
        result2 = workspace.FetchBlob('result2')
        np.testing.assert_array_equal([0, 5, 1, 0, 0], result2)

        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexSize',
            ['index'],
            ['index_size']))
        size = workspace.FetchBlob('index_size')
        self.assertEqual(size, 6)

        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexStore',
            ['index'],
            ['stored_entries']))
        stored_actual = workspace.FetchBlob('stored_entries')
        new_entries = np.array([entries[3], entries[4]], dtype=dtype)
        expected = np.concatenate((my_entries, new_entries))
        if dtype is str:
            # we'll always get bytes back from Caffe2
            expected = np.array([
                x.item().encode('utf-8') if isinstance(x, np.str_) else x
                for x in expected
            ], dtype=object)
        np.testing.assert_array_equal(expected, stored_actual)

        workspace.RunOperatorOnce(core.CreateOperator(
            index_create_op,
            [],
            ['index2']))

        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexLoad',
            ['index2', 'stored_entries'],
            ['index2'],
            skip_first_entry=1))

        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexSize',
            ['index2'],
            ['index2_size']))
        index2_size = workspace.FetchBlob('index2_size')
        self.assertEqual(index2_size, 5)

        # test serde
        with tempfile.NamedTemporaryFile() as tmp:
            workspace.RunOperatorOnce(core.CreateOperator(
                'Save',
                ['index'],
                [],
                absolute_path=1,
                db_type='minidb',
                db=tmp.name))
            # frees up the blob
            workspace.FeedBlob('index', np.array([]))
            # reloads the index
            workspace.RunOperatorOnce(core.CreateOperator(
                'Load',
                [],
                ['index'],
                absolute_path=1,
                db_type='minidb',
                db=tmp.name))
            query3 = np.array(
                [entries[0], entries[3], entries[0], entries[4], entries[4]],
                dtype=dtype)

            workspace.FeedBlob('query3', query3)
            workspace.RunOperatorOnce(core.CreateOperator(
                'IndexGet', ['index', 'query3'], ['result3']))
            result3 = workspace.FetchBlob('result3')
            np.testing.assert_array_equal([1, 4, 1, 5, 5], result3)

    def test_string_index_ops(self):
        self._test_index_ops([
            'entry1', 'entry2', 'entry3', 'new_entry1',
            'new_entry2', 'miss1', 'miss2', 'miss3',
        ], str, 'StringIndexCreate')

    def test_int_index_ops(self):
        self._test_index_ops(list(range(8)), np.int32, 'IntIndexCreate')

    def test_long_index_ops(self):
        self._test_index_ops(list(range(8)), np.int64, 'LongIndexCreate')

if __name__ == "__main__":
    import unittest
    unittest.main()
