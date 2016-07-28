from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np


class TestIndexOps(TestCase):
    def test_index_ops(self):
        workspace.RunOperatorOnce(core.CreateOperator(
            'StringIndexCreate',
            [],
            ['index'],
            max_elements=10))
        entries = np.array(['entry1', 'entry2', 'entry3'], dtype=str)

        workspace.FeedBlob('entries', entries)
        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexLoad',
            ['index', 'entries'],
            []))
        query1 = np.array(['entry1', 'new_entry1', 'entry1', 'new_entry2'],
                          dtype=str)

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
            []))

        query2 = np.array(['miss1', 'new_entry2', 'entry1', 'miss2', 'miss3'],
                          dtype=str)
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
        self.assertEquals(size, 6)

        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexStore',
            ['index'],
            ['stored_entries']))
        stored_actual = workspace.FetchBlob('stored_entries')
        new_entries = np.array(['new_entry1', 'new_entry2'], dtype=str)
        np.testing.assert_array_equal(
            np.concatenate((entries, new_entries)), stored_actual)

        workspace.RunOperatorOnce(core.CreateOperator(
            'StringIndexCreate',
            [],
            ['index2']))

        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexLoad',
            ['index2', 'stored_entries'],
            [],
            skip_first_entry=1))

        workspace.RunOperatorOnce(core.CreateOperator(
            'IndexSize',
            ['index2'],
            ['index2_size']))
        index2_size = workspace.FetchBlob('index2_size')
        self.assertEquals(index2_size, 5)
