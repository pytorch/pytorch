from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace, dataset
from caffe2.python.dataset import Const
from caffe2.python.schema import List, Struct, Scalar, Map
from caffe2.python.test_util import TestCase


def _assert_arrays_equal(actual, ref, err_msg):
    if ref.dtype.kind in ('S', 'O'):
        np.testing.assert_array_equal(actual, ref, err_msg=err_msg)
    else:
        np.testing.assert_allclose(
            actual, ref, atol=1e-4, rtol=1e-4, err_msg=err_msg)


class TestDatasetOps(TestCase):
    def test_dataset_ops(self):
        """
        1. Defining the schema of our dataset.

        This example schema could represent, for example, a search query log.
        """
        schema = Struct(
            # fixed size vector, which will be stored as a matrix when batched
            ('dense', Scalar((np.float32, 3))),
            # could represent a feature map from feature ID to float value
            ('floats', Map(
                Scalar(np.int32),
                Scalar(np.float32))),
            # could represent a multi-valued categorical feature map
            ('int_lists', Map(
                Scalar(np.int32),
                List(Scalar(np.int64)),
            )),
            # could represent a multi-valued, weighted categorical feature map
            ('id_score_pairs', Map(
                Scalar(np.int32),
                Map(
                    Scalar(np.int64),
                    Scalar(np.float32),
                    keys_name='ids',
                    values_name='scores'),
            )),
            # additional scalar information
            ('metadata', Struct(
                ('user_id', Scalar(np.int64)),
                ('user_embed', Scalar((np.float32, 2))),
                ('query', Scalar(str)),
            )),
        )

        """
        This is what the flattened fields for this schema look like, along
        with its type. Each one of these fields will be stored, read and
        writen as a tensor.
        """
        expected_fields = [
            ('dense', (np.float32, 3)),
            ('floats:lengths', np.int32),
            ('floats:values:keys', np.int32),
            ('floats:values:values', np.float32),
            ('int_lists:lengths', np.int32),
            ('int_lists:values:keys', np.int32),
            ('int_lists:values:values:lengths', np.int32),
            ('int_lists:values:values:values', np.int64),
            ('id_score_pairs:lengths', np.int32),
            ('id_score_pairs:values:keys', np.int32),
            ('id_score_pairs:values:values:lengths', np.int32),
            ('id_score_pairs:values:values:values:ids', np.int64),
            ('id_score_pairs:values:values:values:scores', np.float32),
            ('metadata:user_id', np.int64),
            ('metadata:user_embed', (np.float32, 2)),
            ('metadata:query', str),
        ]
        zipped = zip(
            expected_fields,
            schema.field_names(),
            schema.field_types())
        for (ref_name, ref_type), name, dtype in zipped:
            self.assertEquals(ref_name, name)
            self.assertEquals(np.dtype(ref_type), dtype)

        """
        2. The contents of our dataset.

        Contents as defined below could represent, for example, a log of
        search queries along with dense, sparse features and metadata.
        The datset below has 3 top-level entries.
        """
        contents_raw = [
            # dense
            [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]],
            # floats
            [1, 2, 3],  # len
            [11, 21, 22, 31, 32, 33],  # key
            [1.1, 2.1, 2.2, 3.1, 3.2, 3.3],  # value
            # int lists
            [2, 0, 1],  # len
            [11, 12, 31],  # key
            [2, 4, 3],  # value:len
            [111, 112, 121, 122, 123, 124, 311, 312, 313],  # value:value
            # id score pairs
            [1, 2, 2],  # len
            [11, 21, 22, 31, 32],  # key
            [1, 1, 2, 2, 3],  # value:len
            [111, 211, 221, 222, 311, 312, 321, 322, 323],  # value:ids
            [11.1, 21.1, 22.1, 22.2, 31.1, 31.2, 32.1, 32.2, 32.3],  # val:score
            # metadata
            [123, 234, 456],  # user_id
            [[0.2, 0.8], [0.5, 0.5], [0.7, 0.3]],  # user_embed
            ['dog posts', 'friends who like to', 'posts about ca'],  # query
        ]
        # convert the above content to ndarrays, checking against the schema
        contents = dataset.to_ndarray_list(contents_raw, schema)

        """
        3. Creating and appending to the dataset.
        We first create an empty dataset with the given schema.
        Then, a Writer is used to append these entries to the dataset.
        """
        ds = dataset.Dataset(schema)
        net = core.Net('init')
        ds.init_empty(net)

        blobs_to_append = [Const(net, c) for c in contents]
        writer = ds.writer(init_net=net)
        writer.write(net, blobs_to_append)
        workspace.RunNetOnce(net)

        """
        4. Iterating through the dataset contents.

        If we were to iterate through the top level entries of our dataset,
        this is what we should expect to see:
        """
        entries_raw = [
            (
                [[1.1, 1.2, 1.3]],  # dense
                [1], [11], [1.1],  # floats
                [2], [11, 12], [2, 4], [111, 112, 121, 122, 123, 124],  # intlst
                [1], [11], [1], [111], [11.1],  # id score pairs
                [123], [[0.2, 0.8]], ['dog posts'],  # metadata
            ),
            (
                [[2.1, 2.2, 2.3]],  # dense
                [2], [21, 22], [2.1, 2.2],  # floats
                [0], [], [], [],  # int list
                [2], [21, 22], [1, 2], [211, 221, 222], [21.1, 22.1, 22.2],
                [234], [[0.5, 0.5]], ['friends who like to'],  # metadata
            ),
            (
                [[3.1, 3.2, 3.3]],  # dense
                [3], [31, 32, 33], [3.1, 3.2, 3.3],  # floats
                [1], [31], [3], [311, 312, 313],  # int lst
                [2], [31, 32], [2, 3], [311, 312, 321, 322, 323],
                [31.1, 31.2, 32.1, 32.2, 32.3],  # id score list
                [456], [[0.7, 0.3]], ['posts about ca'],  # metadata
            ),
            # after the end of the dataset, we will keep getting empty vectors
            ([],) * 16,
            ([],) * 16,
        ]
        entries = [dataset.to_ndarray_list(e, schema) for e in entries_raw]

        """
        Let's go ahead and create the reading nets.
        We will run `read` net multiple times and assert that we are reading the
        entries the way we stated above.
        """
        read_init_net = core.Net('read_init')
        read_next_net = core.Net('read_next')
        reader = ds.reader(read_init_net)
        should_continue, batch_blobs = reader.read(read_next_net)

        workspace.RunNetOnce(read_init_net)

        workspace.CreateNet(read_next_net)
        read_next_net_name = str(read_next_net)

        for i, entry in enumerate(entries):
            workspace.RunNet(read_next_net_name)
            for name, blob, base in zip(ds.field_names(), batch_blobs, entry):
                data = workspace.FetchBlob(str(blob))
                _assert_arrays_equal(
                    data, base,
                    err_msg='Mismatch in entry %d, field %s' % (i, name))

        """
        5. Reading/writing in a single plan

        If all of operations on the data are expressible as Caffe2 operators,
        we don't need to load the data to python, iterating through the dataset
        in a single Plan.

        Where we will process the dataset a little and store it in a second
        dataset. We can reuse the same Reader since it supports reset.
        """
        reset_net = core.Net('reset_net')
        reader.reset(reset_net)
        read_step, fields = reader.execution_step()

        """ We will add the line number * 1000 to the feature ids. """
        process_net = core.Net('process')
        line_no = Const(process_net, 0, dtype=np.int32)
        const_one = Const(process_net, 1000, dtype=np.int32)
        process_net.Add([line_no, const_one], [line_no])
        fid = schema.floats.values.keys.id()
        process_net.Print(fields[fid], [])
        process_net.Add([fields[fid], line_no], fields[fid], broadcast=1)

        """ Lets create a second dataset and append to it. """
        ds2 = dataset.Dataset(schema, name='dataset2')
        ds2.init_empty(reset_net)
        writer = ds2.writer(reset_net)
        writer.write(process_net, fields)
        # commit is not necessary for DatasetWriter but will add it for
        # generality of the example
        commit_net = core.Net('commit')
        writer.commit(commit_net)

        """ Time to create and run a plan which will do the processing """
        plan = core.Plan('process')
        plan.AddStep(core.execution_step('reset', reset_net))
        plan.AddStep(read_step.AddNet(process_net))
        plan.AddStep(core.execution_step('commit', commit_net))
        workspace.RunPlan(plan)

        """
        Now we should have dataset2 populated.
        """
        ds2blobs = ds2.get_blobs()
        for i, (name, blob) in enumerate(zip(schema.field_names(), ds2blobs)):
            data = workspace.FetchBlob(str(blob))
            content = contents[i]
            if i == fid:
                # one of our fields has been added with line numbers * 1000
                content += [1000, 2000, 2000, 3000, 3000, 3000]
            _assert_arrays_equal(
                data, contents[i], err_msg='Mismatch in field %s.' % name)

        """
        6. Slicing a dataset

        You can create a new schema from pieces of another schema and reuse
        the same data.
        """
        subschema = Struct(('top_level', schema.int_lists.values))
        int_list_contents = contents[schema.int_lists.values.slice()]
        self.assertEquals(len(subschema.field_names()), len(int_list_contents))

        """
        7. Random Access a dataset

        """
        read_init_net = core.Net('read_init')
        read_next_net = core.Net('read_next')

        idx = np.array([2, 1, 0])
        workspace.FeedBlob('idx', idx)

        reader = ds.random_reader(read_init_net, 'idx')
        reader.computeoffset(read_init_net)

        should_continue, batch_blobs = reader.read(read_next_net)

        workspace.CreateNet(read_init_net)
        workspace.RunNetOnce(read_init_net)

        workspace.CreateNet(read_next_net)
        read_next_net_name = str(read_next_net)

        for i in range(len(entries)):
            k = idx[i] if i in idx else i
            entry = entries[k]
            workspace.RunNet(read_next_net_name)
            for name, blob, base in zip(ds.field_names(), batch_blobs, entry):
                data = workspace.FetchBlob(str(blob))
                _assert_arrays_equal(
                    data, base,
                    err_msg='Mismatch in entry %d, field %s' % (i, name))

        """
        8. Sort and shuffle a dataset

        This sort the dataset using the score of a certain column,
        and then shuffle within each chunk of size batch_size * shuffle_size
        before shuffling the chunks.

        """
        read_init_net = core.Net('read_init')
        read_next_net = core.Net('read_next')

        reader = ds.random_reader(read_init_net)
        reader.sortAndShuffle(read_init_net, 'int_lists:lengths', 1, 2)
        reader.computeoffset(read_init_net)

        should_continue, batch_blobs = reader.read(read_next_net)

        workspace.CreateNet(read_init_net)
        workspace.RunNetOnce(read_init_net)

        workspace.CreateNet(read_next_net)
        read_next_net_name = str(read_next_net)

        expected_idx = np.array([2, 1, 0])
        for i in range(len(entries)):
            k = expected_idx[i] if i in expected_idx else i
            entry = entries[k]
            workspace.RunNet(read_next_net_name)
            for name, blob, base in zip(ds.field_names(), batch_blobs, entry):
                data = workspace.FetchBlob(str(blob))
                _assert_arrays_equal(
                    data, base,
                    err_msg='Mismatch in entry %d, field %s' % (i, name))
