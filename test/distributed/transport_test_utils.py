import pickle

import torch


class TransportTestMixin:
    def make_transport_pair(self):
        raise NotImplementedError

    def test_transport_conformance(self) -> None:
        first, second = self.make_transport_pair()
        try:
            source = torch.arange(4096, dtype=torch.int32)
            destination = torch.zeros_like(source)
            source_memory = first.register_memory(source)
            destination_memory = second.register_memory(destination)
            remote = pickle.loads(pickle.dumps(destination_memory.to_remote_buffer()))

            self.assertEqual(source_memory.to_view().size(), source.nbytes)
            self.assertEqual(destination_memory.to_mutable_view(16, 32).size(), 32)
            self.assertEqual(first.write(source_memory.to_view(), remote), 0)
            self.assertEqual(destination, source)

            read_target = torch.zeros_like(source)
            read_memory = first.register_memory(read_target)
            self.assertEqual(
                first.read(
                    read_memory.to_mutable_view(),
                    destination_memory.to_remote_buffer(),
                ),
                0,
            )
            self.assertEqual(read_target, source)
        finally:
            first.close()
            second.close()

    def test_transport_partial_view_and_registration_reuse(self) -> None:
        first, second = self.make_transport_pair()
        try:
            source = torch.arange(64, dtype=torch.uint8)
            destination = torch.zeros(16, dtype=torch.uint8)
            source_memory = first.register_memory(source)
            destination_memory = second.register_memory(destination)

            self.assertFalse(source_memory.reused_registration())
            self.assertTrue(first.register_memory(source).reused_registration())
            self.assertEqual(
                first.write(
                    source_memory.to_view(16, 16),
                    destination_memory.to_remote_buffer(),
                ),
                0,
            )
            self.assertEqual(destination, source[16:32])
            with self.assertRaises(ValueError):
                source_memory.to_view(source.nbytes + 1)
        finally:
            first.close()
            second.close()
