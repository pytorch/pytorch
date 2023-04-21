# Owner(s): ["module: nn"]

import itertools
import random

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import torch.nn.utils.rnn as rnn_utils


class PackedSequenceTest(TestCase):

    _type_by_name = {
        'torch.DoubleTensor': (torch.DoubleTensor, 'double'),
        'torch.FloatTensor': (torch.FloatTensor, 'float'),
        # We leave out `'torch.HalfTensor': (torch.HalfTensor, 'half'),`
        # because of an error in `pad_packed_sequence`
        # > AttributeError: 'torch.HalfTensor' object has no attribute 'fill_'
        'torch.LongTensor': (torch.LongTensor, 'long'),
        'torch.IntTensor': (torch.IntTensor, 'int'),
        'torch.ShortTensor': (torch.ShortTensor, 'short'),
        'torch.CharTensor': (torch.CharTensor, 'char'),
        'torch.ByteTensor': (torch.ByteTensor, 'byte'),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5
        self.max_length = 6

    def _ordered_sequence(self, tensor_type):
        """Create ordered list of random sequences"""
        seqs = [tensor_type(random.randint(1, self.max_length))
                for _ in range(self.batch_size)]
        if tensor_type == torch.ByteTensor:
            seqs = [s.random_(0, 256) for s in seqs]
        else:
            seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    def _padded_sequence(self, tensor_type):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(tensor_type)
        lengths = [len(i) for i in ordered]
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    def test_type_casts(self):
        """Test type casting of `PackedSequence` against type casting of tensor"""
        for _, (input_type, _) in self._type_by_name.items():
            for expected_type_str, (_, cast_str) in self._type_by_name.items():
                for enforce_sorted in [True, False]:
                    padded, lengths = self._padded_sequence(input_type)
                    packed = rnn_utils.pack_padded_sequence(
                        padded, lengths, enforce_sorted=enforce_sorted)
                    # Apply cast to `PackedSequence` instance and unpack
                    masked = getattr(packed, cast_str)()
                    unpacked, lengths_out = rnn_utils.pad_packed_sequence(masked)
                    self.assertEqual(unpacked.type(), expected_type_str)

    def test_wrong_order(self):
        a = torch.ones(25, 300)
        b = torch.ones(22, 300)
        b_a = rnn_utils.pad_sequence([b, a])
        self.assertRaises(
            RuntimeError,
            lambda: rnn_utils.pack_padded_sequence(b_a, [22, 25], enforce_sorted=True))

    def test_pad_sequence_with_tensor_sequences(self):
        seq_tuple_input = torch.nn.utils.rnn.pad_sequence(
            (torch.tensor([[7, 6]]), torch.tensor([[-7, -1]]))
        )
        seq_tensor_input = torch.nn.utils.rnn.pad_sequence(
            torch.tensor([[[7, 6]], [[-7, -1]]])
        )
        self.assertEqual(seq_tuple_input, seq_tensor_input)
        self.assertEqual(seq_tuple_input.shape, torch.Size([1, 2, 2]))

    def test_pad_sequence_with_non_iterable_sequences(self):
        msg = r"Expected iterable for input sequences, but got arg of type"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.nn.utils.rnn.pad_sequence(5)

    def test_total_length(self):
        padded, lengths = self._padded_sequence(torch.FloatTensor)
        max_length = max(lengths)
        packed = rnn_utils.pack_padded_sequence(padded, lengths)
        # test ValueError if total_length < max_length
        for total_length in (-1, 0, max_length - 1):
            for batch_first in (True, False):
                def err_fn():
                    rnn_utils.pad_packed_sequence(packed, batch_first=batch_first,
                                                  total_length=total_length)
            self.assertRaisesRegex(ValueError,
                                   r'Expected total_length to be at least the '
                                   r'length of the longest sequence in input',
                                   err_fn)
        # test that pad_packed_sequence returns results of correct length
        for batch_first in (True, False):
            no_extra_pad, _ = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first)
            for total_length_delta in (0, 1, 8):
                total_length = max_length + total_length_delta
                unpacked, lengths_out = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first,
                                                                      total_length=total_length)
                self.assertEqual(lengths, lengths_out)
                self.assertEqual(unpacked.size(1 if batch_first else 0), total_length)
                if total_length_delta == 0:
                    ref_output = no_extra_pad
                elif batch_first:
                    extra_pad = no_extra_pad.new_zeros(self.batch_size, total_length_delta)
                    ref_output = torch.cat([no_extra_pad, extra_pad], 1)
                else:
                    extra_pad = no_extra_pad.new_zeros(total_length_delta, self.batch_size)
                    ref_output = torch.cat([no_extra_pad, extra_pad], 0)
                self.assertEqual(unpacked, ref_output)

    def test_to(self):
        for enforce_sorted in (True, False):
            padded, lengths = self._padded_sequence(torch.IntTensor)
            a = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted).cpu()

            self.assertIs(a, a.to('cpu'))
            self.assertIs(a, a.cpu())
            self.assertIs(a, a.to('cpu', dtype=torch.int32))
            self.assertEqual(a.long(), a.to(torch.int64))

            if torch.cuda.is_available():
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = a.cuda(device=cuda)
                    self.assertIs(b, b.to(cuda))
                    self.assertIs(b, b.cuda())
                    self.assertEqual(a, b.to('cpu'))
                    self.assertEqual(b, a.to(cuda))
                    self.assertEqual(a, b.to('cpu', dtype=torch.int32))
                    self.assertIs(b, b.to(dtype=torch.int32))
                    self.assertEqual(b.long(), b.to(dtype=torch.int64))

    def test_to_memory_format(self):
        m = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, bias=True)
        m = m.to(memory_format=torch.channels_last)
        for param in m.parameters():
            if param.dim() == 4:
                self.assertTrue(param.is_contiguous(memory_format=torch.channels_last))

    def test_pad_sequence(self):
        def pad(tensor, length):
            return torch.cat(
                [tensor.data, tensor.data.new(
                    length - tensor.size(0), *tensor.size()[1:]).zero_()])

        # single dimensional
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])

        # batch_first = true
        expected = torch.tensor([[4, 5, 0], [1, 2, 3], [6, 0, 0]])
        padded = rnn_utils.pad_sequence([b, a, c], True)
        self.assertEqual(padded, expected)

        # batch_first = false
        padded = rnn_utils.pad_sequence([b, a, c])
        self.assertEqual(padded, expected.transpose(0, 1))

        # pad with non-zero value
        expected = torch.tensor([[4, 5, 1], [1, 2, 3], [6, 1, 1]])
        padded = rnn_utils.pad_sequence([b, a, c], True, 1)
        self.assertEqual(padded, expected)

        # Test pad sorted sequence
        expected = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
        padded = rnn_utils.pad_sequence([a, b, c], True)
        self.assertEqual(padded, expected)

        # more dimensions
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            trailing_dims = [4] * num_dim
            for i in range(1, maxlen + 1):
                seq_len = i * i
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            random.shuffle(sequences)
            expected = []
            for seq in sequences:
                expected.append(pad(seq, maxlen * maxlen))
            # batch first = true
            expected = torch.stack(expected)
            padded = rnn_utils.pad_sequence(sequences, True)
            self.assertEqual(padded, expected)

            # batch first = false
            padded = rnn_utils.pad_sequence(sequences)
            self.assertEqual(padded, expected.transpose(0, 1))

    def test_unpad_sequence(self):

        # single dimensional
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        sequences = [a, b, c]

        lengths = torch.as_tensor([v.size(0) for v in sequences])
        for batch_first in [True, False]:
            padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=batch_first)
            unpadded_sequences = rnn_utils.unpad_sequence(padded_sequences, lengths, batch_first=batch_first)
            self.assertEqual(sequences, unpadded_sequences)

        # more dimensions
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            trailing_dims = [4] * num_dim
            for i in range(1, maxlen + 1):
                seq_len = i * i
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            random.shuffle(sequences)

            lengths = torch.as_tensor([v.size(0) for v in sequences])
            padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=batch_first)
            unpadded_sequences = rnn_utils.unpad_sequence(padded_sequences, lengths, batch_first=batch_first)
            self.assertEqual(sequences, unpadded_sequences)

    def test_pack_sequence(self):
        def _compatibility_test(sequences, lengths, batch_first, enforce_sorted=False):
            padded = rnn_utils.pad_sequence(sequences, batch_first)
            packed = rnn_utils.pack_sequence(sequences, enforce_sorted)
            unpacked = rnn_utils.pad_packed_sequence(packed, batch_first)
            self.assertEqual(padded, unpacked[0])
            pack_padded = rnn_utils.pack_padded_sequence(
                padded, lengths, batch_first, enforce_sorted)
            self.assertEqual(packed, pack_padded)

        # single dimensional
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        packed = rnn_utils.pack_sequence([a, b, c], enforce_sorted=False)
        expected = torch.tensor([1, 4, 6, 2, 5, 3])
        self.assertEqual(packed.batch_sizes, [3, 2, 1])
        self.assertEqual(packed.data.data, expected)
        self.assertEqual(packed.sorted_indices, [0, 1, 2])
        self.assertEqual(packed.unsorted_indices, [0, 1, 2])

        packed_unsorted = rnn_utils.pack_sequence([b, c, a], enforce_sorted=False)
        self.assertEqual(packed_unsorted.batch_sizes, [3, 2, 1])
        self.assertEqual(packed_unsorted.data.data, expected)
        self.assertEqual(packed_unsorted.sorted_indices, [2, 0, 1])
        self.assertEqual(packed_unsorted.unsorted_indices, [1, 2, 0])

        # single dimensional, enforce_sorted = True
        packed_enforce_sorted = rnn_utils.pack_sequence([a, b, c], enforce_sorted=True)
        self.assertEqual(packed_enforce_sorted.batch_sizes, [3, 2, 1])
        self.assertEqual(packed_enforce_sorted.data.data, expected)
        self.assertTrue(packed_enforce_sorted.sorted_indices is None)
        self.assertTrue(packed_enforce_sorted.unsorted_indices is None)

        with self.assertRaisesRegex(RuntimeError, 'must be sorted in decreasing order'):
            rnn_utils.pack_sequence([b, c, a], enforce_sorted=True)

        with self.assertRaisesRegex(RuntimeError, 'You can pass `enforce_sorted=False`'):
            rnn_utils.pack_sequence([b, c, a], enforce_sorted=True)

        # more dimensions
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            lengths = []
            trailing_dims = [4] * num_dim
            for i in range(maxlen, 0, -1):
                seq_len = i * i
                lengths.append(seq_len)
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            unsorted_sequences = [s.clone() for s in sequences]
            random.shuffle(unsorted_sequences)
            unsorted_sequences_lengths = [t.size(0) for t in unsorted_sequences]

            # compatibility with other utilities
            for batch_first in (True, False):
                for enforce_sorted in (True, False):
                    _compatibility_test(sequences, lengths, batch_first, enforce_sorted)
                _compatibility_test(unsorted_sequences, unsorted_sequences_lengths,
                                    batch_first)

    def test_unpack_sequence(self):

        # single dimensional
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        sequences = [a, b, c]

        packed_sequences = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
        unpacked_sequences = rnn_utils.unpack_sequence(packed_sequences)
        self.assertEqual(sequences, unpacked_sequences)

        # more dimensions
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            trailing_dims = [4] * num_dim
            for i in range(1, maxlen + 1):
                seq_len = i * i
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            random.shuffle(sequences)

            packed_sequences = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
            unpacked_sequences = rnn_utils.unpack_sequence(packed_sequences)
            self.assertEqual(sequences, unpacked_sequences)

    def test_pack_padded_sequence(self):
        def generate_test_case(sorted_lengths, should_shuffle):
            def pad(tensor, length):
                return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

            max_length = sorted_lengths[0]
            batch_sizes = [sum(map(bool, filter(lambda x: x >= i, sorted_lengths)))
                           for i in range(1, max_length + 1)]
            offset = 0
            padded = torch.cat([pad(i * 100 + torch.arange(1., 5 * l + 1).view(l, 1, 5), max_length)
                                for i, l in enumerate(sorted_lengths, 1)], 1)
            expected_data = [[torch.arange(1., 6) + (i + 1) * 100 + 5 * n for i in range(batch_size)]
                             for n, batch_size in enumerate(batch_sizes)]
            expected_data = list(itertools.chain.from_iterable(expected_data))
            expected_data = torch.stack(expected_data, dim=0)

            if should_shuffle:
                # Shuffle the padded sequence to create an unsorted sequence
                permutation = list(range(len(sorted_lengths)))
                random.shuffle(permutation)

                unsorted_indices = torch.tensor(permutation)
                padded = padded.index_select(1, unsorted_indices)
                lengths = torch.tensor(sorted_lengths).index_select(0, unsorted_indices)
            else:
                unsorted_indices = None
                lengths = sorted_lengths

            return padded.requires_grad_(), lengths, expected_data, batch_sizes, unsorted_indices

        test_cases = [
            # sorted_lengths, should_shuffle
            [[10, 8, 4, 2, 2, 2, 1], False],
            [[11, 10, 8, 6, 4, 3, 1], False],
            [[11, 10, 8, 6, 4, 3, 1], True],
        ]

        for test_case, batch_first in itertools.product(test_cases, (True, False)):
            sorted_lengths, should_shuffle = test_case
            padded, lengths, expected_data, batch_sizes, unsorted_indices = generate_test_case(
                sorted_lengths, should_shuffle)

            src = padded
            if batch_first:
                src = src.transpose(0, 1)

            # check output
            packed = rnn_utils.pack_padded_sequence(src, lengths, batch_first=batch_first,
                                                    enforce_sorted=not should_shuffle)
            self.assertEqual(packed.data.data, expected_data)
            self.assertEqual(packed.batch_sizes, batch_sizes)
            self.assertEqual(packed.unsorted_indices, unsorted_indices)

            # test inverse
            unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first)
            self.assertEqual(unpacked, src)
            self.assertEqual(unpacked_len, lengths)

            # check grad
            if padded.grad is not None:
                padded.grad.data.zero_()
            grad_output = unpacked.data.clone().normal_()
            unpacked.backward(grad_output)
            if batch_first:
                grad_output.transpose_(0, 1)
            for i, l in enumerate(lengths):
                self.assertEqual(padded.grad.data[:l, i], grad_output[:l, i])
                if l < 10:
                    self.assertEqual(padded.grad.data[l:, i].abs().sum(), 0)

        # test error messages
        with self.assertRaisesRegex(RuntimeError, 'You can pass `enforce_sorted=False`'):
            packed = rnn_utils.pack_padded_sequence(torch.randn(3, 3), [1, 3, 2])
        with self.assertRaisesRegex(RuntimeError, 'empty tensor'):
            packed = rnn_utils.pack_padded_sequence(torch.randn(0, 0), [])
        with self.assertRaisesRegex(RuntimeError, 'empty tensor'):
            packed = rnn_utils.pack_padded_sequence(torch.randn([0, 1, 10]),
                                                    torch.randn([11, 14, 14, 2]), True)


if __name__ == '__main__':
    run_tests()
