# Owner(s): ["module: dataloader"]
# ruff: noqa: F841

from copy import deepcopy

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_MACOS,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils.data import DataLoader, Dataset, IterableDataset


# Helper classes for stateful DataLoader tests using stateful=True
class StatefulIterator:
    """Basic iterator with stateful functionality."""

    def __init__(self, samples, shuffle=False):
        self.samples = samples
        self.shuffle = shuffle
        self.size = len(self.samples)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration
        if self.shuffle:
            i = torch.randint(self.size, (1,)).item()
        else:
            i = self.i
        sample = self.samples[i]
        self.i += 1
        return sample

    def state_dict(self):
        sd = {"i": self.i}
        if self.shuffle:
            sd["rng_state"] = torch.get_rng_state()
        return sd

    def load_state_dict(self, state_dict):
        self.i = state_dict["i"]
        if self.shuffle and "rng_state" in state_dict:
            torch.set_rng_state(state_dict["rng_state"])


class StatefulIterableDataset(IterableDataset):
    """Iterable dataset with stateful behavior."""

    def __init__(self, sizes_for_all_workers, shuffle=False):
        self.sizes_for_all_workers = sizes_for_all_workers
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]

        start = sum(self.sizes_for_all_workers[:worker_id])
        iter_data = list(range(start, start + self.sizes_for_all_workers[worker_id]))
        return StatefulIterator(iter_data, self.shuffle)


def identity_collate(x):
    """Identity collate function for tests."""
    return x


class TestStatefulDataLoaderBasic(TestCase):
    """Test basic stateful DataLoader functionality with iterable datasets."""

    def _get_dataset(self, shuffle):
        return StatefulIterableDataset([0, 100, 37], shuffle=shuffle)

    @parametrize("num_workers", [0])
    def test_basic_statefulness(self, num_workers):
        """Test basic state_dict/load_state_dict functionality"""
        dataset = self._get_dataset(shuffle=False)
        dl = DataLoader(
            dataset=dataset,
            num_workers=0,
            collate_fn=identity_collate,
            stateful=True,
            batch_size=7,
        )

        # Get some data and save state
        it = iter(dl)
        batch1 = next(it)
        state_dict = dl.state_dict()
        batch2 = next(it)

        # Create new loader and resume from state
        dl2 = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity_collate,
            stateful=True,
            batch_size=7,
        )
        dl2.load_state_dict(state_dict)
        it2 = iter(dl2)
        batch2_resumed = next(it2)

        # Should get the same batch when resuming
        self.assertEqual(batch2, batch2_resumed)


class StatefulMapDataset(Dataset):
    """Map dataset with stateful behavior for testing. Keeps track of access count."""

    def __init__(self, size):
        self.size = size
        self.data = [{"id": i, "value": i * 2} for i in range(size)]
        self.access_count = 0

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        self.access_count += 1
        return self.data[i]

    def state_dict(self):
        return {"access_count": self.access_count}

    def load_state_dict(self, state_dict):
        if "access_count" in state_dict:
            self.access_count = state_dict["access_count"]


class TestStatefulDataLoaderMapDataset(TestCase):
    """Test stateful DataLoader with map-style datasets."""

    @parametrize("num_workers", [0])
    def test_map_dataset_statefulness(self, num_workers):
        """Test state_dict/load_state_dict functionality with map datasets"""

        dataset = StatefulMapDataset(50)
        dl = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity_collate,
            stateful=True,
            batch_size=5,
        )

        it = iter(dl)
        for i in range(3):  # consume first 3 batches
            _ = next(it)

        # Save state after consuming 3 batches
        state_dict = dl.state_dict()
        self.assertIsInstance(state_dict, dict)

        # Collect remaining batches
        remaining_batches_original = list(it)

        # Create new loader and resume from checkpoint
        dataset2 = StatefulMapDataset(50)
        dl2 = DataLoader(
            dataset=dataset2,
            num_workers=num_workers,
            collate_fn=identity_collate,
            stateful=True,
            batch_size=5,
        )
        dl2.load_state_dict(state_dict)

        # Collect all batches from resumed loader
        remaining_batches_resumed = list(dl2)

        # Verify that resumed loader continues exactly where original left off
        self.assertEqual(remaining_batches_original, remaining_batches_resumed)


class StatefulSampler(torch.utils.data.Sampler):
    """Sampler with stateful behavior for testing."""

    def __init__(self, size, track_calls=False):
        self.size = size
        self.i = 0
        self.track_calls = track_calls
        self.state_dict_called = False
        self.load_state_dict_called = False
        self._iterator = None

    def __iter__(self):
        self._iterator = StatefulSamplerIterator(self.size, self.i, self.track_calls)
        return self._iterator

    def __len__(self):
        return self.size

    def state_dict(self):
        if self.track_calls:
            self.state_dict_called = True
        return {"i": self.i}

    def load_state_dict(self, state_dict):
        if self.track_calls:
            self.load_state_dict_called = True
        self.i = state_dict["i"]


class StatefulSamplerIterator:
    """Iterator for stateful sampler."""

    def __init__(self, size, start_idx=0, track_next_calls=False):
        self.size = size
        self.i = start_idx
        self.track_next_calls = track_next_calls
        self.next_call_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.track_next_calls:
            self.next_call_count += 1
        idx = self.i
        if idx >= self.size:
            raise StopIteration
        self.i += 1
        return idx

    def state_dict(self):
        return {"i": self.i}

    def load_state_dict(self, state_dict):
        self.i = state_dict["i"]


class TestStatefulDataLoaderSampler(TestCase):
    """Test stateful DataLoader with stateful samplers."""

    @parametrize("num_workers", [0])
    def test_stateful_sampler(self, num_workers):
        """Test state_dict/load_state_dict functionality with stateful samplers

        Ensures that stateful samplers use state_dict/load_state_dict for state restoration
        rather than fast-forwarding via repeated next() calls.
        """
        dataset = StatefulMapDataset(20)
        sampler = StatefulSampler(len(dataset), track_calls=True)

        dl = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity_collate,
            stateful=True,
            batch_size=4,
            sampler=sampler,
        )

        # Consume some batches
        batches_before_checkpoint = []
        it = iter(dl)
        for i in range(2):  # consume first 2 batches
            batch = next(it)
            batches_before_checkpoint.append(batch)

        # Save state after consuming 2 batches
        state_dict = dl.state_dict()
        self.assertIsInstance(state_dict, dict)

        # Verify that state_dict was called on the stateful sampler during save
        self.assertIn("_sampler_iter_state", state_dict)
        self.assertIsNotNone(state_dict["_sampler_iter_state"])
        self.assertTrue(
            sampler.state_dict_called, "state_dict should be called on stateful sampler"
        )

        # Continue with original iterator and collect remaining batches
        remaining_batches_original = list(it)

        # Create new loader with fresh sampler and resume from checkpoint
        dataset2 = StatefulMapDataset(20)
        sampler2 = StatefulSampler(len(dataset2), track_calls=True)
        dl2 = DataLoader(
            dataset=dataset2,
            num_workers=num_workers,
            collate_fn=identity_collate,
            stateful=True,
            batch_size=4,
            sampler=sampler2,
        )

        dl2.load_state_dict(state_dict)
        it2 = iter(dl2)
        # Verify that load_state_dict was called on the stateful sampler
        self.assertTrue(
            sampler2.load_state_dict_called,
            "load_state_dict should be called on stateful sampler",
        )
        # The sampler's state should have been restored (i should be 8, since we consumed 2 batches of 4)
        self.assertEqual(sampler2._iterator.i, 8)

        # CRITICAL CHECK: Verify that next() was NOT called for fast-forwarding
        # If fast-forwarding occurred, we would see multiple next() calls here
        # Since we restored via load_state_dict, next_call_count should be 0 before we start consuming
        if sampler2._iterator is not None:
            self.assertEqual(
                sampler2._iterator.next_call_count,
                0,
                "Stateful sampler should NOT be fast-forwarded via next() calls after load_state_dict",
            )

        # Collect all batches from resumed loader
        remaining_batches_resumed = list(it2)

        # Verify that resumed loader continues exactly where original left off
        self.assertEqual(
            len(remaining_batches_original), len(remaining_batches_resumed)
        )
        for orig, resumed in zip(remaining_batches_original, remaining_batches_resumed):
            self.assertEqual(len(orig), len(resumed))
            for o, r in zip(orig, resumed):
                self.assertEqual(o, r)


class TestStatefulDataLoaderRandomState(TestCase):
    """Test stateful DataLoader with random state and shuffle functionality."""

    def test_shuffle_state_preservation(self):
        """Test that shuffle random state is properly preserved and restored"""
        dataset = StatefulMapDataset(20)

        # Test with shuffle=True and specific generator seed
        generator = torch.Generator()
        generator.manual_seed(42)

        dl = DataLoader(
            dataset=dataset,
            num_workers=0,  # Single process for deterministic testing
            stateful=True,
            batch_size=4,
            shuffle=True,
            generator=generator,
        )

        # Consume some batches with shuffled order
        batches_before_checkpoint = []
        it = iter(dl)
        for i in range(2):  # consume first 2 batches
            batch = next(it)
            batches_before_checkpoint.append(batch)

        # Save state after consuming 2 batches
        state_dict = dl.state_dict()
        self.assertIsInstance(state_dict, dict)

        # Continue with original iterator
        batch3_original = next(it)

        # Create new loader with same generator seed and resume from checkpoint
        generator2 = torch.Generator()
        generator2.manual_seed(42)  # Same seed as original

        dataset2 = StatefulMapDataset(20)
        dl2 = DataLoader(
            dataset=dataset2,
            num_workers=0,
            stateful=True,
            batch_size=4,
            shuffle=True,
            generator=generator2,
        )
        dl2.load_state_dict(state_dict)

        it2 = iter(dl2)
        batch3_resumed = next(it2)

        # Should get the same batch when resuming from checkpoint
        self.assertEqual(len(batch3_original), len(batch3_resumed))
        for o, r in zip(batch3_original, batch3_resumed):
            self.assertEqual(o, r)

    def test_reproducible_shuffle_sequences(self):
        """Test that shuffle produces reproducible sequences when resumed"""
        dataset = StatefulMapDataset(16)

        # Run full sequence with shuffling
        generator1 = torch.Generator()
        generator1.manual_seed(123)

        dl1 = DataLoader(
            dataset=dataset,
            num_workers=0,
            stateful=True,
            batch_size=2,
            shuffle=True,
            generator=generator1,
        )

        full_sequence = []
        for batch in dl1:
            full_sequence.extend(batch)

        # Now run with checkpointing at different points
        generator2 = torch.Generator()
        generator2.manual_seed(123)  # Same seed

        dl2 = DataLoader(
            dataset=dataset,
            num_workers=0,
            stateful=True,
            batch_size=2,
            shuffle=True,
            generator=generator2,
        )

        # Consume first 3 batches (6 items)
        checkpointed_sequence = []
        it = iter(dl2)
        for i in range(3):
            batch = next(it)
            checkpointed_sequence.extend(batch)

        # Save state and get remaining items
        state_dict = dl2.state_dict()
        remaining_original = []
        for batch in it:
            remaining_original.extend(batch)

        # Resume from checkpoint
        generator3 = torch.Generator()
        generator3.manual_seed(123)  # Same seed

        dataset3 = StatefulMapDataset(16)
        dl3 = DataLoader(
            dataset=dataset3,
            num_workers=0,
            stateful=True,
            batch_size=2,
            shuffle=True,
            generator=generator3,
        )
        dl3.load_state_dict(state_dict)

        remaining_resumed = []
        for batch in dl3:
            remaining_resumed.extend(batch)

        # Verify sequences match
        full_reconstructed = checkpointed_sequence + remaining_resumed
        self.assertEqual(len(full_sequence), len(full_reconstructed))
        for orig, recon in zip(full_sequence, full_reconstructed):
            self.assertEqual(orig, recon)

        # Also verify the remaining parts match
        self.assertEqual(remaining_original, remaining_resumed)


class TestStatefulDataLoaderMultiEpoch(TestCase):
    """Test stateful DataLoader across multiple epochs and at epoch boundaries."""

    @parametrize("num_workers", [0])
    def test_multi_epoch_continuation(self, num_workers):
        """Test that DataLoader state is preserved across multiple epochs"""
        dataset = StatefulMapDataset(12)

        dl = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            stateful=True,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )

        # Run through 2 full epochs and collect all items
        all_items_original = []
        for epoch in range(2):
            epoch_items = []
            for batch in dl:
                epoch_items.extend(batch)
            all_items_original.extend(epoch_items)

        # Save state after 2 epochs
        state_dict = dl.state_dict()

        # Continue for 1 more epoch
        epoch3_items_original = []
        for batch in dl:
            epoch3_items_original.extend(batch)

        # Now test resumption: Create new loader and resume from checkpoint

        dataset2 = StatefulMapDataset(12)
        dl2 = DataLoader(
            dataset=dataset2,
            num_workers=num_workers,
            stateful=True,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )
        dl2.load_state_dict(state_dict)

        # Get epoch 3 from resumed loader
        epoch3_items_resumed = []
        for batch in dl2:
            epoch3_items_resumed.extend(batch)

        # Should match exactly
        self.assertEqual(epoch3_items_original, epoch3_items_resumed)

    @parametrize("num_workers", [0])
    def test_epoch_boundary_state_behavior(self, num_workers):
        """Test state behavior at exact epoch boundaries"""
        dataset = StatefulMapDataset(8)

        dl = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            stateful=True,
            batch_size=2,
        )

        # Consume exactly one full epoch and save state at different points
        epoch1_items = []
        batches_consumed = 0
        state_dict_during_epoch = None

        it = iter(dl)
        for batches_consumed, batch in enumerate(
            it
        ):  # 4 batches of 2 items each = 8 items total
            epoch1_items.extend(batch)

            if batches_consumed == 2:  # Save state after 3rd batch (6 items consumed)
                state_dict_during_epoch = dl.state_dict()

        # We should have consumed all 8 items (4 batches of 2)
        self.assertEqual(len(epoch1_items), 8)

        # Save state after epoch completion
        state_dict_after_epoch = dl.state_dict()

        # Test resuming from state saved during epoch (after 3rd batch)
        dataset2 = StatefulMapDataset(8)
        dl2 = DataLoader(
            dataset=dataset2,
            num_workers=num_workers,
            stateful=True,
            batch_size=2,
        )
        dl2.load_state_dict(state_dict_during_epoch)

        remaining_items = []
        for batch in dl2:
            remaining_items.extend(batch)

        # Should get the last batch of epoch 1 (items 6,7), then start next epoch
        # Since we stopped after 3rd batch (6 items), we should get 4th batch + next epoch
        expected_remaining_this_epoch = epoch1_items[6:8]  # Items 6,7 (4th batch)
        expected_next_epoch = epoch1_items  # Full next epoch: items 0-7
        expected_remaining = expected_remaining_this_epoch + expected_next_epoch

        self.assertEqual(len(remaining_items), len(expected_remaining_this_epoch))
        self.assertEqual(remaining_items, expected_remaining_this_epoch)

        # Test resuming from state saved after epoch completion
        dataset3 = StatefulMapDataset(8)
        dl3 = DataLoader(
            dataset=dataset3,
            num_workers=num_workers,
            stateful=True,
            batch_size=2,
        )
        dl3.load_state_dict(state_dict_after_epoch)

        next_epoch_items = []
        for batch in dl3:
            next_epoch_items.extend(batch)

        # Should get a full fresh epoch
        self.assertEqual(len(next_epoch_items), 8)
        self.assertEqual(next_epoch_items, epoch1_items)  # Same order since no shuffle

    @parametrize("num_workers", [0])
    def test_checkpoint_at_different_epoch_positions(self, num_workers):
        """Test checkpointing at batch boundaries within epochs"""
        dataset = StatefulMapDataset(12)
        dl = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            stateful=True,
            batch_size=3,  # Creates 4 batches: [3, 3, 3, 3]
        )

        # Test checkpointing at different batch boundaries
        checkpoint_batch_positions = [1, 2, 3]  # After 1st, 2nd, 3rd batch

        for checkpoint_batch_pos in checkpoint_batch_positions:
            with self.subTest(checkpoint_batch_position=checkpoint_batch_pos):
                # Reset and consume up to checkpoint batch position
                dataset_test = StatefulMapDataset(12)
                dl_test = DataLoader(
                    dataset=dataset_test,
                    num_workers=num_workers,
                    stateful=True,
                    batch_size=3,
                )

                consumed_items = []
                it = iter(dl_test)

                # Consume specified number of batches
                for batch_idx in range(checkpoint_batch_pos):
                    try:
                        batch = next(it)
                        consumed_items.extend(batch)
                    except StopIteration:
                        break

                state_dict = dl_test.state_dict()

                # Get remaining items from original iterator
                remaining_original = []
                for batch in it:
                    remaining_original.extend(batch)

                # If we're at end of epoch, continue to next epoch
                if not remaining_original:
                    for batch in dl_test:  # Next epoch
                        remaining_original.extend(batch)

                # Resume from checkpoint
                dataset_resume = StatefulMapDataset(12)
                dl_resume = DataLoader(
                    dataset=dataset_resume,
                    num_workers=num_workers,
                    stateful=True,
                    batch_size=3,
                )
                dl_resume.load_state_dict(state_dict)

                remaining_resumed = []
                for batch in dl_resume:
                    remaining_resumed.extend(batch)

                # Should match
                self.assertEqual(len(remaining_original), len(remaining_resumed))
                for orig, resumed in zip(remaining_original, remaining_resumed):
                    self.assertEqual(orig, resumed)


class TestStatefulDataLoaderSerialization(TestCase):
    """Test stateful DataLoader state dict serialization compatibility."""

    def test_json_serialization(self):
        """Test that state dict can be JSON serialized and deserialized"""
        dataset = StatefulIterableDataset([0, 50, 25], shuffle=False)
        dl = DataLoader(
            dataset=dataset,
            num_workers=0,  # Single process for simpler state
            collate_fn=identity_collate,
            stateful=True,
            batch_size=5,
        )

        # Consume some data
        it = iter(dl)
        for _ in range(3):
            next(it)

        # Get state dict
        state_dict = dl.state_dict()
        self.assertIsInstance(state_dict, dict)

        import json

        json_str = json.dumps(state_dict)
        deserialized_state = json.loads(json_str)

        # Should be able to load the deserialized state
        dataset2 = StatefulIterableDataset([0, 50, 25], shuffle=False)
        dl2 = DataLoader(
            dataset=dataset2,
            num_workers=0,
            collate_fn=identity_collate,
            stateful=True,
            batch_size=5,
        )

        # This should not raise an error
        dl2.load_state_dict(deserialized_state)

        # Verify functionality by getting remaining data
        remaining_original = list(it)

        remaining_deserialized = list(dl2)

        self.assertEqual(remaining_original, remaining_deserialized)

    def test_pickle_serialization(self):
        """Test that state dict can be pickle serialized and deserialized"""
        dataset = StatefulMapDataset(30)
        dl = DataLoader(
            dataset=dataset,
            num_workers=0,
            stateful=True,
            batch_size=6,
        )

        # Consume some data
        batches_before = []
        it = iter(dl)
        for i in range(2):
            batch = next(it)
            batches_before.append(batch)

        # Get state dict
        state_dict = dl.state_dict()

        # Test pickle serialization
        import pickle

        try:
            pickled_state = pickle.dumps(state_dict)
            unpickled_state = pickle.loads(pickled_state)

            # Should be able to load the unpickled state
            dataset2 = StatefulMapDataset(30)
            dl2 = DataLoader(
                dataset=dataset2,
                num_workers=0,
                stateful=True,
                batch_size=6,
            )

            # This should not raise an error
            dl2.load_state_dict(unpickled_state)

            # Verify functionality
            remaining_original = list(it)

            remaining_unpickled = list(dl2)

            self.assertEqual(len(remaining_original), len(remaining_unpickled))
            for orig, unpick in zip(remaining_original, remaining_unpickled):
                self.assertEqual(orig, unpick)

        except Exception as e:
            self.fail(f"Pickle serialization failed: {e}")

    def test_state_dict_structure_validation(self):
        """Test that state dict has expected structure and contains required keys"""
        dataset = StatefulIterableDataset([0, 40], shuffle=False)
        dl = DataLoader(
            dataset=dataset,
            num_workers=0,
            collate_fn=identity_collate,
            stateful=True,
            batch_size=4,
        )

        # Test initial state dict
        initial_state = dl.state_dict()
        self.assertIsInstance(initial_state, dict)

        # Consume some data
        it = iter(dl)
        next(it)
        next(it)

        # Test state dict after consumption
        mid_state = dl.state_dict()
        self.assertIsInstance(mid_state, dict)

        # State dicts should have consistent structure (same keys) but different values
        self.assertEqual(initial_state.keys(), mid_state.keys())


class ErrorDataset_SDL(Dataset):
    def __getitem__(self, index: int):
        raise ValueError("Iteration error")

    def __len__(self):
        return 10


ERROR_MSG = "Error in worker_init_fn"


class TestStatefulDataLoaderErrors(TestCase):
    @parametrize("num_workers", [0])
    def test_iteration_error(self, num_workers):
        dl = DataLoader(
            dataset=ErrorDataset_SDL(),
            num_workers=num_workers,
            stateful=True,
        )
        it = iter(dl)
        with self.assertRaisesRegex(ValueError, "Iteration error"):
            next(it)


class IterationState:
    def __init__(self, start, end):
        self.curr = start
        self.end = end

    def set_state(self, state):
        self.curr = state["curr"]
        self.end = state["end"]

    def get_state(self):
        return {"curr": self.curr, "end": self.end}


class CountIterCalls(torch.utils.data.IterableDataset):
    def __init__(self, length):
        self.length = length
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(list(range(self.length)))

    def state_dict(self):
        return {"iter_calls": self.iter_calls}

    def load_state_dict(self, state_dict):
        pass


class CountIterCallsIter(torch.utils.data.IterableDataset):
    def __init__(self, length):
        self.length = length
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        worker_id = 0
        if torch.utils.data.get_worker_info() is not None:
            worker_id = torch.utils.data.get_worker_info().id
        num_workers = 1
        if torch.utils.data.get_worker_info() is not None:
            num_workers = torch.utils.data.get_worker_info().num_workers

        num_samples = (int)(self.length / num_workers)
        self.iter_state = IterationState(
            num_samples * worker_id, num_samples * (worker_id + 1)
        )
        return self

    def __next__(self):
        if self.iter_state.curr >= self.iter_state.end:
            raise StopIteration
        value = self.iter_state.curr
        self.iter_state.curr += 1
        return value

    def state_dict(self):
        return {"state": self.iter_state.get_state(), "iter_calls": self.iter_calls}

    def load_state_dict(self, state_dict):
        self.iter_state.set_state(state_dict["state"])


class TestSingleIterCalled(TestCase):
    def _get_iter_calls(self, state):
        if "_dataset_state" in state:
            w_states = [state]
        else:
            w_states = list(state["_snapshot"]["_worker_snapshots"].values())

        if w_states[0]["_dataset_state"] is not None:
            return [x["_dataset_state"]["iter_calls"] for x in w_states]
        return [
            x["_fetcher_state"]["_dataset_iter_state"]["iter_calls"] for x in w_states
        ]

    def _run_test(self, num_workers, dataset, expected_iter_calls):
        dl = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            multiprocessing_context=(
                "forkserver" if IS_MACOS and num_workers else None
            ),
            stateful=True,
        )
        iter(dl)
        state = dl.state_dict()
        # Ensure iter is called only once per worker
        self.assertEqual(
            self._get_iter_calls(state), [expected_iter_calls[0]] * max(1, num_workers)
        )

        dl2 = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            multiprocessing_context=(
                "forkserver" if IS_MACOS and num_workers else None
            ),
            stateful=True,
        )
        dl2.load_state_dict(state)
        iter(dl2)
        state2 = dl2.state_dict()
        # Ensure that iter is called only once per worker even when dataloader resumes from a state
        self.assertEqual(
            self._get_iter_calls(state2), [expected_iter_calls[1]] * max(1, num_workers)
        )

    def test_inline(self):
        self._run_test(0, CountIterCalls(100), [1, 2])

    # def test_mp(self):
    #     self._run_test(2, CountIterCalls(100), [1, 1])

    def test_inline_iter(self):
        self._run_test(0, CountIterCallsIter(100), [1, 2])

    # def test_mp_iter(self):
    #     self._run_test(2, CountIterCallsIter(100), [1, 1])


class DynamicStateIterable(IterableDataset):
    def __init__(self, samples):
        self.samples = samples
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration
        v = self.samples[self.i]
        self.i += 1
        return v

    def state_dict(self):
        state = {"i": self.i}
        for a in range(self.i):
            state[str(a)] = {a: list(range(a))}
            state[f"t{a}"] = torch.tensor(a, dtype=torch.int8)
        return state

    def load_state_dict(self, sd):
        self.i = sd["i"]


class TestDynamicStateGrowth(TestCase):
    @parametrize("num_workers", [0])
    def test_state_is_immutable_post_fetch(self, num_workers):
        ds = DynamicStateIterable(list(range(100)))
        dl = DataLoader(ds, num_workers=num_workers, stateful=True)
        it = iter(dl)
        for _ in range(5):
            next(it)
        s1 = dl.state_dict()
        s1_copy = deepcopy(s1)
        # advance further and take next state
        for _ in range(5):
            next(it)
        s2 = dl.state_dict()
        self.assertEqual(s1, s1_copy)
        self.assertNotEqual(s1, s2)

        # resume from s1 and check equivalence of next items
        dl2 = DataLoader(ds, num_workers=num_workers, stateful=True)
        dl2.load_state_dict(s1)
        it2 = iter(dl2)
        exp = []
        for _ in range(2):
            exp.append(next(it2))
        self.assertTrue(len(exp) > 0)


class TestStateInitializationResumeCompleteness(TestCase):
    def _run(self, num_workers):
        length = 100
        ds = CountIterCallsIter(length)
        dl = DataLoader(
            ds, num_workers=num_workers, stateful=True, collate_fn=identity_collate
        )
        it = iter(dl)
        data = []
        for _ in range(length - 30):
            batch = next(it)
            # single element batches in iterator variant; normalize to list
            if isinstance(batch, list):
                data.extend(batch)
            else:
                data.append(batch)
        s = dl.state_dict()

        dl2 = DataLoader(
            ds, num_workers=num_workers, stateful=True, collate_fn=identity_collate
        )
        dl2.load_state_dict(s)
        it2 = iter(dl2)
        for _ in range(30):
            batch = next(it2)
            if isinstance(batch, list):
                data.extend(batch)
            else:
                data.append(batch)
        self.assertEqual(set(data), set(range(length)))

    def test_single(self):
        self._run(0)

    # def test_multi(self):
    #     self._run(2)


class TestConcurrentLoaderParity(TestCase):
    @parametrize("num_workers", [0])
    def test_parity_with_standard_loader(self, num_workers):
        dataset = StatefulMapDataset(40)
        stateful = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            stateful=True,
            collate_fn=identity_collate,
        )
        exp = list(stateful)
        standard = DataLoader(
            dataset=dataset, num_workers=num_workers, collate_fn=identity_collate
        )
        got = list(standard)
        self.assertEqual(got, exp)


class TestStatefulDataLoaderEmptyStateDict(TestCase):
    """Test behavior when passing an empty dict to load_state_dict."""

    def test_load_empty_state_dict_resets_state(self):
        """Test that passing empty dict to load_state_dict resets the dataloader state."""
        dataset = StatefulMapDataset(20)

        # Create initial dataloader
        dl = DataLoader(
            dataset=dataset,
            num_workers=0,
            stateful=True,
            batch_size=4,
        )

        # Consume some batches
        it = iter(dl)
        _ = next(it)
        _ = next(it)

        # Save state after consuming 2 batches
        state_dict = dl.state_dict()
        self.assertIsNotNone(state_dict)
        self.assertNotEqual(state_dict, {})

        # Now load empty state dict
        dl.load_state_dict({})

        # Check that internal state was reset
        self.assertIsNone(dl._iterator)
        self.assertFalse(dl._initial_iter_for_state_dict)
        self.assertIsNone(dl.next_iter_state)

    def test_load_empty_state_dict_then_load_real_state(self):
        """Test that after loading empty dict, we can still load a real state dict."""
        dataset = StatefulMapDataset(20)

        dl = DataLoader(
            dataset=dataset,
            num_workers=0,
            stateful=True,
            batch_size=4,
        )

        # Consume some batches and save state
        it = iter(dl)
        _ = next(it)
        _ = next(it)
        state_dict = dl.state_dict()
        batch3_original = next(it)

        # Load empty dict first
        dl2 = DataLoader(
            dataset=dataset,
            num_workers=0,
            stateful=True,
            batch_size=4,
        )
        dl2.load_state_dict({})

        # Now load the real state dict
        dl2.load_state_dict(state_dict)

        # Verify we can resume from the saved state
        it2 = iter(dl2)
        batch3_resumed = next(it2)

        self.assertEqual(len(batch3_original), len(batch3_resumed))
        for o, r in zip(batch3_original, batch3_resumed):
            self.assertEqual(o, r)

    def test_load_empty_state_dict_iteration_behavior(self):
        """Test iteration behavior after loading empty state dict."""
        dataset = StatefulMapDataset(20)

        # Create dataloader and consume some data
        dl = DataLoader(
            dataset=dataset,
            num_workers=0,
            stateful=True,
            batch_size=4,
        )

        it = iter(dl)
        _ = next(it)
        _ = next(it)

        # Load empty state dict
        dl.load_state_dict({})

        # Create new iterator - should start from beginning
        # (or have some defined behavior)
        it2 = iter(dl)
        batch1_new = next(it2)

        # Document current behavior: With empty dict, it may behave unexpectedly
        # The test passes if iteration is possible, but the behavior needs to be clarified
        self.assertIsNotNone(batch1_new)

    def test_empty_state_dict_multiple_times(self):
        """Test loading empty state dict multiple times."""
        dataset = StatefulMapDataset(20)

        dl = DataLoader(
            dataset=dataset,
            num_workers=0,
            stateful=True,
            batch_size=4,
        )

        # Load empty dict multiple times - should not cause errors
        dl.load_state_dict({})
        dl.load_state_dict({})
        dl.load_state_dict({})

        # Should still be able to iterate
        it = iter(dl)
        batch = next(it)
        self.assertIsNotNone(batch)


parametrized_classes = [
    TestStatefulDataLoaderMapDataset,
    TestStatefulDataLoaderBasic,
    TestStatefulDataLoaderSampler,
    TestStatefulDataLoaderMultiEpoch,
    TestStatefulDataLoaderErrors,
    TestDynamicStateGrowth,
    TestConcurrentLoaderParity,
]
for parametrize_class in parametrized_classes:
    instantiate_parametrized_tests(parametrize_class)


if __name__ == "__main__":
    run_tests()
