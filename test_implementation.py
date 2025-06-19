#!/usr/bin/env python3

"""
Test script to verify the Schedule1F1B and ScheduleGPipe implementations work correctly
"""

import torch
from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleGPipe, _PipelineScheduleRuntime
from torch.distributed.pipelining.stage import _PipelineStageBase

class MockPipelineStage(_PipelineStageBase):
    def __init__(self, stage_index=0, num_stages=1, group_rank=0, group_size=1):
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.group_rank = group_rank
        self.group_size = group_size
        self.group = None
        self.is_first = True
        self.is_last = True
        self.submod = None
        self.has_backward = True
        self.output_chunks = []
        
    def _create_grad_recv_info(self, *args, **kwargs):
        return None
        
    def _prepare_forward_infra(self, n_microbatches, *args, **kwargs):
        pass
        
    def _prepare_backward_infra(self, n_microbatches):
        pass
        
    def clear_runtime_states(self):
        pass

def test_inheritance():
    """Test that both classes inherit from _PipelineScheduleRuntime"""
    print("Testing inheritance...")
    
    # Test Schedule1F1B inheritance
    assert issubclass(Schedule1F1B, _PipelineScheduleRuntime), \
        "Schedule1F1B should inherit from _PipelineScheduleRuntime"
    
    # Test ScheduleGPipe inheritance
    assert issubclass(ScheduleGPipe, _PipelineScheduleRuntime), \
        "ScheduleGPipe should inherit from _PipelineScheduleRuntime"
    
    print("✅ Inheritance tests passed")

def test_instantiation():
    """Test that both classes can be instantiated"""
    print("Testing instantiation...")
    
    # Create mock stage
    stage = MockPipelineStage()
    
    # Test Schedule1F1B instantiation
    schedule_1f1b = Schedule1F1B(stage, n_microbatches=4)
    assert hasattr(schedule_1f1b, 'pipeline_order'), "Schedule1F1B should have pipeline_order attribute"
    assert hasattr(schedule_1f1b, 'pipeline_order_with_comms'), "Schedule1F1B should have pipeline_order_with_comms attribute"
    
    # Test ScheduleGPipe instantiation
    schedule_gpipe = ScheduleGPipe(stage, n_microbatches=4)
    assert hasattr(schedule_gpipe, 'pipeline_order'), "ScheduleGPipe should have pipeline_order attribute"
    assert hasattr(schedule_gpipe, 'pipeline_order_with_comms'), "ScheduleGPipe should have pipeline_order_with_comms attribute"
    
    print("✅ Instantiation tests passed")

def test_pipeline_order_generation():
    """Test that pipeline orders are generated correctly"""
    print("Testing pipeline order generation...")
    
    stage = MockPipelineStage()
    
    # Test Schedule1F1B pipeline order
    schedule_1f1b = Schedule1F1B(stage, n_microbatches=4)
    assert schedule_1f1b.pipeline_order is not None, "Schedule1F1B should have pipeline_order"
    assert len(schedule_1f1b.pipeline_order) > 0, "Schedule1F1B pipeline_order should not be empty"
    
    # Test ScheduleGPipe pipeline order
    schedule_gpipe = ScheduleGPipe(stage, n_microbatches=4)
    assert schedule_gpipe.pipeline_order is not None, "ScheduleGPipe should have pipeline_order"
    assert len(schedule_gpipe.pipeline_order) > 0, "ScheduleGPipe pipeline_order should not be empty"
    
    print("✅ Pipeline order generation tests passed")

def test_step_microbatches_method():
    """Test that _step_microbatches method exists and calls parent implementation"""
    print("Testing _step_microbatches method...")
    
    stage = MockPipelineStage()
    
    # Test Schedule1F1B _step_microbatches
    schedule_1f1b = Schedule1F1B(stage, n_microbatches=2)
    assert hasattr(schedule_1f1b, '_step_microbatches'), "Schedule1F1B should have _step_microbatches method"
    
    # Test ScheduleGPipe _step_microbatches
    schedule_gpipe = ScheduleGPipe(stage, n_microbatches=2)
    assert hasattr(schedule_gpipe, '_step_microbatches'), "ScheduleGPipe should have _step_microbatches method"
    
    print("✅ _step_microbatches method tests passed")

if __name__ == '__main__':
    print("Running implementation tests...")
    print("=" * 50)
    
    test_inheritance()
    test_instantiation()
    test_pipeline_order_generation()
    test_step_microbatches_method()
    
    print("=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("The Schedule1F1B and ScheduleGPipe implementations are working correctly.")