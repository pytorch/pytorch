from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace, dataset
from caffe2.python.dataset import Const
from caffe2.python.schema import (
    List, Field, Struct, Scalar, Map, from_blob_list, FetchRecord, NewRecord,
    FeedRecord
)
from caffe2.python.test_util import TestCase

import numpy.testing as npt

import string

from hypothesis import given
import hypothesis.strategies as st


def _assert_arrays_equal(actual, ref, err_msg):
    if ref.dtype.kind in ('S', 'O', 'U'):
        np.testing.assert_array_equal(actual, ref, err_msg=err_msg)
    else:
        np.testing.assert_allclose(
            actual, ref, atol=1e-4,
            rtol=1e-4, err_msg=err_msg
        )


def _assert_records_equal(actual, ref):
    assert isinstance(actual, Field)
    assert isinstance(ref, Field)
    b1 = actual.field_blobs()
    b2 = ref.field_blobs()
    assert (len(b1) == len(b2)), 'Records have different lengths: %d vs. %d' % (
        len(b1), len(b2)
    )
    for name, d1, d2 in zip(ref.field_names(), b1, b2):
        _assert_arrays_equal(d1, d2, err_msg='Mismatch in field %s.' % name)


@st.composite
def _sparse_features_map(draw, num_records, **kwargs):
    sparse_maps_lengths = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=num_records,
            max_size=num_records
        )
    )

    sparse_maps_total_length = sum(sparse_maps_lengths)

    sparse_keys = draw(
        st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=sparse_maps_total_length,
            max_size=sparse_maps_total_length,
            unique=True
        )
    )

    sparse_values_lengths = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=sparse_maps_total_length,
            max_size=sparse_maps_total_length
        )
    )

    total_sparse_values_lengths = sum(sparse_values_lengths)

    sparse_values = draw(
        # max_value is max int64
        st.lists(
            st.integers(min_value=1, max_value=9223372036854775807),
            min_size=total_sparse_values_lengths,
            max_size=total_sparse_values_lengths
        )
    )

    return [
        sparse_maps_lengths,
        sparse_keys,
        sparse_values_lengths,
        sparse_values,
    ]


@st.composite
def _dense_features_map(draw, num_records, **kwargs):
    float_lengths = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=num_records,
            max_size=num_records
        )
    )

    total_length = sum(float_lengths)

    float_keys = draw(
        st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=total_length,
            max_size=total_length,
            unique=True
        )
    )

    float_values = draw(
        st.lists(st.floats(),
                 min_size=total_length,
                 max_size=total_length)
    )

    return [float_lengths, float_keys, float_values]


@st.composite
def _dataset(draw, min_elements=3, max_elements=10, **kwargs):
    schema = Struct(
        # Dense Features Map
        ('floats', Map(
            Scalar(np.int32), Scalar(np.float32)
        )),
        # Sparse Features Map
        ('int_lists', Map(
            Scalar(np.int32),
            List(Scalar(np.int64)),
        )),
        # Complex Type
        ('text', Scalar(str)),
    )

    num_records = draw(
        st.integers(min_value=min_elements,
                    max_value=max_elements)
    )

    raw_dense_features_map_contents = draw(_dense_features_map(num_records))

    raw_sparse_features_map_contents = draw(_sparse_features_map(num_records))

    raw_text_contents = [
        draw(
            st.lists(
                st.text(alphabet=string.ascii_lowercase),
                min_size=num_records,
                max_size=num_records
            )
        )
    ]

    # Concatenate all raw contents to a single one
    contents_raw = raw_dense_features_map_contents + raw_sparse_features_map_contents + raw_text_contents

    contents = from_blob_list(schema, contents_raw)

    return (schema, contents, num_records)


class TestDatasetOps(TestCase):
    @given(_dataset())
    def test_pack_unpack(self, input):
        """
        Tests if packing and unpacking of the whole dataset is an identity.
        """
        (schema, contents, num_records) = input

        dataset_fields = schema.field_names()

        net = core.Net('pack_unpack_net')

        batch = NewRecord(net, contents)
        FeedRecord(batch, contents)

        packed = net.PackRecords(
            batch.field_blobs(), 1,
            fields=dataset_fields
        )

        unpacked = packed.UnPackRecords(
            [], len(dataset_fields),
            fields=dataset_fields
        )

        workspace.RunNetOnce(net)

        for initial_tensor, unpacked_tensor in zip(
            batch.field_blobs(), unpacked
        ):
            npt.assert_array_equal(
                workspace.FetchBlob(initial_tensor),
                workspace.FetchBlob(unpacked_tensor)
            )

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
                Scalar(np.int32), Scalar(np.float32)
            )),
            # could represent a multi-valued categorical feature map
            ('int_lists', Map(
                Scalar(np.int32),
                List(Scalar(np.int64)),
            )),
            # could represent a multi-valued, weighted categorical feature map
            (
                'id_score_pairs', Map(
                    Scalar(np.int32),
                    Map(
                        Scalar(np.int64),
                        Scalar(np.float32),
                        keys_name='ids',
                        values_name='scores'
                    ),
                )
            ),
            # additional scalar information
            (
                'metadata', Struct(
                    ('user_id', Scalar(np.int64)),
                    ('user_embed', Scalar((np.float32, 2))),
                    ('query', Scalar(str)),
                )
            ),
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
            expected_fields, schema.field_names(), schema.field_types()
        )
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
        contents = from_blob_list(schema, contents_raw)
        """
        3. Creating and appending to the dataset.
        We first create an empty dataset with the given schema.
        Then, a Writer is used to append these entries to the dataset.
        """
        ds = dataset.Dataset(schema)
        net = core.Net('init')
        with core.NameScope('init'):
            ds.init_empty(net)

            content_blobs = NewRecord(net, contents)
            FeedRecord(content_blobs, contents)
            writer = ds.writer(init_net=net)
            writer.write_record(net, content_blobs)
        workspace.RunNetOnce(net)
        """
        4. Iterating through the dataset contents.

        If we were to iterate through the top level entries of our dataset,
        this is what we should expect to see:
        """
        entries_raw = [
            (
                [[1.1, 1.2, 1.3]],  # dense
                [1],
                [11],
                [1.1],  # floats
                [2],
                [11, 12],
                [2, 4],
                [111, 112, 121, 122, 123, 124],  # intlst
                [1],
                [11],
                [1],
                [111],
                [11.1],  # id score pairs
                [123],
                [[0.2, 0.8]],
                ['dog posts'],  # metadata
            ),
            (
                [[2.1, 2.2, 2.3]],  # dense
                [2],
                [21, 22],
                [2.1, 2.2],  # floats
                [0],
                [],
                [],
                [],  # int list
                [2],
                [21, 22],
                [1, 2],
                [211, 221, 222],
                [21.1, 22.1, 22.2],
                [234],
                [[0.5, 0.5]],
                ['friends who like to'],  # metadata
            ),
            (
                [[3.1, 3.2, 3.3]],  # dense
                [3],
                [31, 32, 33],
                [3.1, 3.2, 3.3],  # floats
                [1],
                [31],
                [3],
                [311, 312, 313],  # int lst
                [2],
                [31, 32],
                [2, 3],
                [311, 312, 321, 322, 323],
                [31.1, 31.2, 32.1, 32.2, 32.3],  # id score list
                [456],
                [[0.7, 0.3]],
                ['posts about ca'],  # metadata
            ),
            # after the end of the dataset, we will keep getting empty vectors
            ([], ) * 16,
            ([], ) * 16,
        ]
        entries = [from_blob_list(schema, e) for e in entries_raw]
        """
        Let's go ahead and create the reading nets.
        We will run `read` net multiple times and assert that we are reading the
        entries the way we stated above.
        """
        read_init_net = core.Net('read_init')
        read_next_net = core.Net('read_next')
        reader = ds.reader(read_init_net)
        should_continue, batch = reader.read_record(read_next_net)

        workspace.RunNetOnce(read_init_net)
        workspace.CreateNet(read_next_net, True)

        for entry in entries:
            workspace.RunNet(str(read_next_net))
            actual = FetchRecord(batch)
            _assert_records_equal(actual, entry)
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
        read_step, batch = reader.execution_step()
        """ We will add the line number * 1000 to the feature ids. """
        process_net = core.Net('process')
        line_no = Const(process_net, 0, dtype=np.int32)
        const_one = Const(process_net, 1000, dtype=np.int32)
        process_net.Add([line_no, const_one], [line_no])
        field = batch.floats.keys.get()
        process_net.Print(field, [])
        process_net.Add([field, line_no], field, broadcast=1, axis=0)
        """ Lets create a second dataset and append to it. """
        ds2 = dataset.Dataset(schema, name='dataset2')
        ds2.init_empty(reset_net)
        writer = ds2.writer(reset_net)
        writer.write_record(process_net, batch)
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
        ds2_data = FetchRecord(ds2.content())
        field = ds2_data.floats.keys
        field.set(blob=field.get() - [1000, 2000, 2000, 3000, 3000, 3000])
        _assert_records_equal(contents, ds2_data)
        """
        6. Slicing a dataset

        You can create a new schema from pieces of another schema and reuse
        the same data.
        """
        subschema = Struct(('top_level', schema.int_lists.values))
        int_list_contents = contents.int_lists.values.field_names()
        self.assertEquals(len(subschema.field_names()), len(int_list_contents))
        """
        7. Random Access a dataset

        """
        read_init_net = core.Net('read_init')
        read_next_net = core.Net('read_next')

        idx = np.array([2, 1, 0])
        indices_blob = Const(read_init_net, idx, name='indices')
        reader = ds.random_reader(read_init_net, indices_blob)
        reader.computeoffset(read_init_net)

        should_stop, batch = reader.read_record(read_next_net)

        workspace.CreateNet(read_init_net, True)
        workspace.RunNetOnce(read_init_net)

        workspace.CreateNet(read_next_net, True)

        for i in range(len(entries)):
            k = idx[i] if i in idx else i
            entry = entries[k]
            workspace.RunNet(str(read_next_net))
            actual = FetchRecord(batch)
            _assert_records_equal(actual, entry)
        workspace.RunNet(str(read_next_net))
        self.assertEquals(True, workspace.FetchBlob(should_stop))
        """
        8. Random Access a dataset with loop_over = true

        """
        read_init_net = core.Net('read_init')
        read_next_net = core.Net('read_next')

        idx = np.array([2, 1, 0])
        indices_blob = Const(read_init_net, idx, name='indices')
        reader = ds.random_reader(read_init_net, indices_blob, loop_over=True)
        reader.computeoffset(read_init_net)

        should_stop, batch = reader.read_record(read_next_net)

        workspace.CreateNet(read_init_net, True)
        workspace.RunNetOnce(read_init_net)

        workspace.CreateNet(read_next_net, True)

        for _ in range(len(entries) * 3):
            workspace.RunNet(str(read_next_net))
            self.assertEquals(False, workspace.FetchBlob(should_stop))
        """
        9. Sort and shuffle a dataset

        This sort the dataset using the score of a certain column,
        and then shuffle within each chunk of size batch_size * shuffle_size
        before shuffling the chunks.

        """
        read_init_net = core.Net('read_init')
        read_next_net = core.Net('read_next')

        reader = ds.random_reader(read_init_net)
        reader.sort_and_shuffle(read_init_net, 'int_lists:lengths', 1, 2)
        reader.computeoffset(read_init_net)

        should_continue, batch = reader.read_record(read_next_net)

        workspace.CreateNet(read_init_net, True)
        workspace.RunNetOnce(read_init_net)

        workspace.CreateNet(read_next_net, True)

        expected_idx = np.array([2, 1, 0])
        for i in range(len(entries)):
            k = expected_idx[i] if i in expected_idx else i
            entry = entries[k]
            workspace.RunNet(str(read_next_net))
            actual = FetchRecord(batch)
            _assert_records_equal(actual, entry)

    def test_last_n_window_ops(self):
        collect_net = core.Net('collect_net')
        collect_net.GivenTensorFill(
            [],
            'input',
            shape=[3, 2],
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        input_array =\
            np.array(list(range(1, 7)), dtype=np.float32).reshape(3, 2)

        workspace.CreateBlob('output')
        workspace.FeedBlob('next', np.array(0, dtype=np.int32))
        collect_net.LastNWindowCollector(
            ['output', 'next', 'input'],
            ['output', 'next'],
            num_to_collect=7,
        )
        plan = core.Plan('collect_data')
        plan.AddStep(
            core.execution_step('collect_data', [collect_net],
                                num_iter=1)
        )
        workspace.RunPlan(plan)
        reference_result = workspace.FetchBlob('output')
        npt.assert_array_equal(input_array, reference_result)

        plan = core.Plan('collect_data')
        plan.AddStep(
            core.execution_step('collect_data', [collect_net],
                                num_iter=2)
        )
        workspace.RunPlan(plan)
        reference_result = workspace.FetchBlob('output')
        npt.assert_array_equal(input_array[[1, 2, 2, 0, 1, 2, 0]],
                               reference_result)

        plan = core.Plan('collect_data')
        plan.AddStep(
            core.execution_step('collect_data', [collect_net],
                                num_iter=3)
        )
        workspace.RunPlan(plan)
        reference_result = workspace.FetchBlob('output')
        npt.assert_array_equal(input_array[[2, 0, 1, 2, 2, 0, 1]],
                               reference_result)

    def test_collect_tensor_ops(self):
        init_net = core.Net('init_net')
        blobs = ['blob_1', 'blob_2', 'blob_3']
        bvec_map = {}
        ONE = init_net.ConstantFill([], 'ONE', shape=[1, 2], value=1)
        for b in blobs:
            init_net.ConstantFill([], [b], shape=[1, 2], value=0)
            bvec_map[b] = b + '_vec'
            init_net.CreateTensorVector([], [bvec_map[b]])

        reader_net = core.Net('reader_net')
        for b in blobs:
            reader_net.Add([b, ONE], [b])

        collect_net = core.Net('collect_net')
        num_to_collect = 1000
        max_example_to_cover = 100000
        bvec = [bvec_map[b] for b in blobs]
        collect_net.CollectTensor(
            bvec + blobs,
            bvec,
            num_to_collect=num_to_collect,
        )

        print('Collect Net Proto: {}'.format(collect_net.Proto()))

        plan = core.Plan('collect_data')
        plan.AddStep(core.execution_step('collect_init', init_net))
        plan.AddStep(
            core.execution_step(
                'collect_data', [reader_net, collect_net],
                num_iter=max_example_to_cover
            )
        )
        workspace.RunPlan(plan)

        # concat the collected tensors
        concat_net = core.Net('concat_net')
        bconcated_map = {}
        bsize_map = {}
        for b in blobs:
            bconcated_map[b] = b + '_concated'
            bsize_map[b] = b + '_size'
            concat_net.ConcatTensorVector([bvec_map[b]], [bconcated_map[b]])
            concat_net.TensorVectorSize([bvec_map[b]], [bsize_map[b]])

        workspace.RunNetOnce(concat_net)

        # check data
        reference_result = workspace.FetchBlob(bconcated_map[blobs[0]])
        self.assertEqual(
            reference_result.shape,
            (min(num_to_collect, max_example_to_cover), 2)
        )
        size = workspace.FetchBlob(bsize_map[blobs[0]])
        self.assertEqual(tuple(), size.shape)
        self.assertEqual(min(num_to_collect, max_example_to_cover), size.item())

        hist, _ = np.histogram(
            reference_result[:, 0],
            bins=10,
            range=(1, max_example_to_cover)
        )
        print('Sample histogram: {}'.format(hist))

        self.assertTrue(all(hist > 0.6 * (num_to_collect / 10)))
        for i in range(1, len(blobs)):
            result = workspace.FetchBlob(bconcated_map[blobs[i]])
            self.assertEqual(reference_result.tolist(), result.tolist())


if __name__ == "__main__":
    import unittest
    unittest.main()
