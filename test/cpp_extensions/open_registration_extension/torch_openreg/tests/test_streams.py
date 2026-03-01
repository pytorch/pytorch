# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestStream(TestCase):
    @skipIfTorchDynamo()
    def test_stream_create(self):
        """Test stream creation with different methods"""
        stream = torch.Stream(device="openreg")
        self.assertEqual(stream.device_index, torch.openreg.current_device())
        stream = torch.Stream(device="openreg:1")
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream = torch.Stream(1)
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream1 = torch.Stream(
            stream_id=stream.stream_id,
            device_type=stream.device_type,
            device_index=stream.device_index,
        )
        self.assertEqual(stream, stream1)

    @skipIfTorchDynamo()
    def test_stream_context(self):
        """Test stream context manager"""
        with torch.Stream(device="openreg:1") as stream:
            self.assertEqual(torch.accelerator.current_stream(), stream)

    def test_stream_context_exception_restore(self):
        prev = torch.accelerator.current_stream()
        inner_stream = torch.Stream(device="openreg:1")
        try:
            with inner_stream:
                # inside the context we should be on the inner stream
                self.assertEqual(torch.accelerator.current_stream(), inner_stream)
                raise RuntimeError("forced")
        except RuntimeError:
            pass
        # After the exception, the current stream should be restored.
        self.assertEqual(torch.accelerator.current_stream(), prev)

    @skipIfTorchDynamo()
    def test_stream_switch(self):
        """Test switching between streams"""
        stream1 = torch.Stream(device="openreg:0")
        torch.accelerator.set_stream(stream1)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)

        stream2 = torch.Stream(device="openreg:1")
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)
        torch.accelerator.set_stream(stream2)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream2)

    @skipIfTorchDynamo()
    def test_stream_synchronize(self):
        """Test stream synchronization"""
        stream = torch.Stream(device="openreg:1")
        self.assertEqual(True, stream.query())

        event = torch.Event()
        event.record(stream)
        stream.synchronize()
        self.assertEqual(True, stream.query())

    @skipIfTorchDynamo()
    def test_stream_repr(self):
        """Test stream string representation"""
        stream = torch.Stream(device="openreg:1")
        self.assertTrue(
            "torch.Stream device_type=openreg, device_index=1" in repr(stream)
        )

    @skipIfTorchDynamo()
    def test_stream_wait_stream(self):
        """Test stream waiting on another stream"""
        stream_1 = torch.Stream(device="openreg:0")
        stream_2 = torch.Stream(device="openreg:1")
        stream_2.wait_stream(stream_1)

    @skipIfTorchDynamo()
    def test_stream_wait_event(self):
        """Test stream waiting on event"""
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e = s1.record_event()
        s2.wait_event(e)

    @skipIfTorchDynamo()
    def test_stream_equality(self):
        """Test stream equality comparison"""
        stream1 = torch.Stream(device="openreg:0")
        stream2 = torch.Stream(device="openreg:0")

        # Different streams should not be equal
        self.assertNotEqual(stream1, stream2)

        # Same stream should be equal to itself
        self.assertEqual(stream1, stream1)

        # Stream created with same parameters should be equal
        stream3 = torch.Stream(
            stream_id=stream1.stream_id,
            device_type=stream1.device_type,
            device_index=stream1.device_index,
        )
        self.assertEqual(stream1, stream3)

    @skipIfTorchDynamo()
    def test_stream_multiple_devices(self):
        """Test streams on multiple devices"""
        stream0 = torch.Stream(device="openreg:0")
        stream1 = torch.Stream(device="openreg:1")

        self.assertEqual(stream0.device_index, 0)
        self.assertEqual(stream1.device_index, 1)

        # Set current stream for each device
        torch.accelerator.set_device_index(0)
        torch.accelerator.set_stream(stream0)
        self.assertEqual(torch.accelerator.current_stream(), stream0)

        torch.accelerator.set_device_index(1)
        torch.accelerator.set_stream(stream1)
        self.assertEqual(torch.accelerator.current_stream(), stream1)

    @skipIfTorchDynamo()
    def test_stream_context_nested(self):
        """Test nested stream contexts"""
        stream1 = torch.Stream(device="openreg:0")
        stream2 = torch.Stream(device="openreg:0")

        with stream1:
            self.assertEqual(torch.accelerator.current_stream(), stream1)
            with stream2:
                self.assertEqual(torch.accelerator.current_stream(), stream2)
            # Should restore to stream1
            self.assertEqual(torch.accelerator.current_stream(), stream1)

    @skipIfTorchDynamo()
    def test_stream_record_event(self):
        """Test recording events on streams"""
        stream = torch.Stream(device="openreg")
        event = stream.record_event()

        self.assertIsNotNone(event)
        self.assertEqual(event.device.type, "openreg")
        stream.synchronize()
        self.assertTrue(event.query())

    @skipIfTorchDynamo()
    def test_stream_fan_out(self):
        """Test fan-out pattern: one stream to multiple streams"""
        stream_producer = torch.Stream(device="openreg:0")
        stream_consumer1 = torch.Stream(device="openreg:0")
        stream_consumer2 = torch.Stream(device="openreg:0")

        x = torch.randn(40, 40, device="openreg:0")
        y = torch.randn(40, 40, device="openreg:0")

        # producer creates data
        with stream_producer:
            z = torch.matmul(x, y)
            event_producer = stream_producer.record_event()

        # consumers wait for producer
        stream_consumer1.wait_event(event_producer)
        stream_consumer2.wait_event(event_producer)

        # consumers process data
        with stream_consumer1:
            result1 = torch.sum(z)
        with stream_consumer2:
            result2 = torch.sum(z)

        # synchronize all streams
        torch.accelerator.synchronize()

        # verify results
        self.assertTrue(event_producer.query())
        self.assertEqual(result1, result2)  # Same computation should yield same result

    @skipIfTorchDynamo()
    def test_stream_fan_in(self):
        """Test fan-in pattern: multiple streams to one stream"""
        stream1 = torch.Stream(device="openreg:0")
        stream2 = torch.Stream(device="openreg:0")
        stream_aggregator = torch.Stream(device="openreg:0")

        a = torch.randn(30, 30, device="openreg:0")
        b = torch.randn(30, 30, device="openreg:0")

        # multiple producers
        with stream1:
            prod1 = torch.matmul(a, b)
            event1 = stream1.record_event()
        with stream2:
            prod2 = torch.matmul(a, b)
            event2 = stream2.record_event()

        # aggregator waits for all producers
        stream_aggregator.wait_event(event1)
        stream_aggregator.wait_event(event2)

        # aggregator combines results
        with stream_aggregator:
            _ = prod1 + prod2

        torch.accelerator.synchronize()

        # Verify events completed and result correctness
        self.assertTrue(event1.query())
        self.assertTrue(event2.query())

    @skipIfTorchDynamo()
    def test_stream_pipeline(self):
        """Test pipeline pattern: sequential streams"""
        stage1 = torch.Stream(device="openreg:0")
        stage2 = torch.Stream(device="openreg:0")
        stage3 = torch.Stream(device="openreg:0")

        data = torch.randn(50, 50, device="openreg:0")

        # stage 1
        with stage1:
            intermediate1 = torch.matmul(data, data)
            event_stage1 = stage1.record_event()

        # stage 2 waits for stage 1
        stage2.wait_event(event_stage1)
        with stage2:
            intermediate2 = torch.matmul(intermediate1, data)
            event_stage2 = stage2.record_event()

        # stage 3 waits for stage 2
        stage3.wait_event(event_stage2)
        with stage3:
            _ = torch.matmul(intermediate2, data)

        torch.accelerator.synchronize()

        # Verify events completed and result correctness
        self.assertTrue(event_stage1.query())
        self.assertTrue(event_stage2.query())

    @skipIfTorchDynamo()
    def test_complex_stream_dependencies(self):
        """Test complex stream dependency scenarios"""
        # scenario 1: diamond dependencies (A -> B, A -> C, then B and C -> D)
        stream_a = torch.Stream(device="openreg:0")
        stream_b = torch.Stream(device="openreg:0")
        stream_c = torch.Stream(device="openreg:0")
        stream_d = torch.Stream(device="openreg:0")

        data = torch.randn(40, 40, device="openreg:0")

        # A produces data
        with stream_a:
            shared_data = torch.matmul(data, data)
            event_a = stream_a.record_event()

        # B and C both wait for A (diamond start)
        stream_b.wait_event(event_a)
        stream_c.wait_event(event_a)

        with stream_b:
            b_result = torch.matmul(shared_data, data)
            event_b = stream_b.record_event()
        with stream_c:
            c_result = torch.matmul(shared_data, data)
            event_c = stream_c.record_event()

        # D waits for both B and C (diamond end)
        stream_d.wait_event(event_b)
        stream_d.wait_event(event_c)

        with stream_d:
            d_result = b_result + c_result

        torch.accelerator.synchronize()

        # Verify events completed and result correctness
        self.assertTrue(event_a.query())
        self.assertTrue(event_b.query())
        self.assertTrue(event_c.query())


if __name__ == "__main__":
    run_tests()
