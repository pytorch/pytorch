#!/usr/bin/env python3
"""
Test script for custom user metadata in CUDA memory allocations.

This script demonstrates how to:
1. Set custom metadata for memory allocations
2. Allocate tensors with different metadata tags
3. Take a memory snapshot
4. Display the metadata in the snapshot
"""

import torch
import pickle


def test_memory_metadata():
    """Test the custom memory metadata feature."""

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    print("Starting memory metadata test...")
    print()

    # Enable memory history recording
    print("1. Enabling memory history recording...")
    torch.cuda.memory._record_memory_history(enabled="all")

    # Test 1: Set metadata and allocate tensor
    print("2. Setting metadata to 'training_phase'...")
    torch.cuda.memory._set_memory_metadata("training_phase")

    # Verify the metadata was set
    current_metadata = torch.cuda.memory._get_memory_metadata()
    print(f"   Current metadata: '{current_metadata}'")
    assert current_metadata == "training_phase", "Metadata not set correctly!"

    # Allocate a tensor
    print("3. Allocating tensor x (100x100)...")
    x = torch.randn(100, 100, device='cuda')

    # Test 2: Change metadata and allocate another tensor
    print("4. Setting metadata to 'validation_phase'...")
    torch.cuda.memory._set_memory_metadata("validation_phase")

    current_metadata = torch.cuda.memory._get_memory_metadata()
    print(f"   Current metadata: '{current_metadata}'")

    print("5. Allocating tensor y (200x200)...")
    y = torch.randn(200, 200, device='cuda')

    # Test 3: Clear metadata and allocate another tensor
    print("6. Clearing metadata (setting to empty string)...")
    torch.cuda.memory._set_memory_metadata("")

    current_metadata = torch.cuda.memory._get_memory_metadata()
    print(f"   Current metadata: '{current_metadata}'")
    assert current_metadata == "", "Metadata not cleared!"

    print("7. Allocating tensor z (50x50) with no metadata...")
    z = torch.randn(50, 50, device='cuda')

    # Test 4: Take a snapshot
    print("8. Taking memory snapshot...")
    snapshot = torch.cuda.memory._snapshot()

    # Analyze the snapshot
    print()
    print("=" * 70)
    print("SNAPSHOT ANALYSIS")
    print("=" * 70)

    # Look at device traces
    if 'device_traces' in snapshot:
        device_traces = snapshot['device_traces']
        print(f"Number of devices: {len(device_traces)}")

        if len(device_traces) > 0:
            traces = device_traces[0]  # First device
            print(f"Number of trace entries: {len(traces)}")
            print()

            # Find allocation entries with metadata
            alloc_count = 0
            metadata_found = {}

            for i, trace in enumerate(traces):
                if trace.get('action') == 'alloc':
                    alloc_count += 1
                    user_metadata = trace.get('user_metadata', '')
                    size = trace.get('size', 0)

                    if user_metadata:
                        if user_metadata not in metadata_found:
                            metadata_found[user_metadata] = []
                        metadata_found[user_metadata].append({
                            'index': i,
                            'size': size,
                            'addr': trace.get('addr', 'N/A')
                        })

            print(f"Total allocations found: {alloc_count}")
            print()

            # Display allocations grouped by metadata
            if metadata_found:
                print("Allocations with metadata:")
                print("-" * 70)
                for metadata, allocs in metadata_found.items():
                    print(f"\nMetadata: '{metadata}'")
                    for alloc in allocs:
                        print(f"  - Trace #{alloc['index']}: "
                              f"size={alloc['size']:,} bytes, "
                              f"addr=0x{alloc['addr']:x}")
            else:
                print("WARNING: No allocations with metadata found!")
                print("This might indicate the feature is not working correctly.")

            # Show some example traces with metadata
            print()
            print("Sample trace entries with user_metadata field:")
            print("-" * 70)
            shown = 0
            for i, trace in enumerate(traces):
                if 'user_metadata' in trace and trace.get('action') == 'alloc':
                    print(f"\nTrace #{i}:")
                    print(f"  action: {trace.get('action')}")
                    print(f"  size: {trace.get('size'):,} bytes")
                    print(f"  user_metadata: '{trace.get('user_metadata')}'")
                    print(f"  compile_context: '{trace.get('compile_context')}'")
                    shown += 1
                    if shown >= 5:  # Show first 5
                        break

    print()
    print("=" * 70)

    # Save snapshot for inspection
    snapshot_file = "/tmp/memory_snapshot_with_metadata.pickle"
    print(f"9. Saving snapshot to {snapshot_file}...")
    torch.cuda.memory._dump_snapshot(snapshot_file)
    print(f"   Snapshot saved! You can inspect it with:")
    print(f"   python -c \"import pickle; s=pickle.load(open('{snapshot_file}','rb')); print(s)\"")

    # Cleanup
    print()
    print("10. Cleaning up...")
    del x, y, z
    torch.cuda.empty_cache()

    print()
    print("✓ Test completed successfully!")
    print()
    print("Summary:")
    print("  - Successfully set and retrieved custom metadata")
    print("  - Metadata was recorded in memory allocations")
    print("  - Snapshot contains user_metadata field in trace entries")


if __name__ == "__main__":
    test_memory_metadata()
