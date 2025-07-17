import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

class TestExpandableSegments(TestCase):
    def test_expandable_segments(self):
        a = []
        b = []
        c = []

        torch.xpu.memory._set_allocator_settings('expandable_segments:True')
        total = torch.xpu.get_device_properties().total_memory
        loop_times = total // (64 * 1024 * 1024 * torch.float.itemsize)
        loop_times -= 5  # leave some space for sycl runtime
        for i in range(loop_times):
            a.append(torch.ones(64, 1024 * 1024, dtype=torch.float, device="xpu"))

        del a

        for i in range(loop_times):
            b.append(torch.ones(48, 1024 * 1024, dtype=torch.float, device="xpu"))

        for i in range(10):
            c.append(torch.ones(32, 1024 * 1024, dtype=torch.float, device="xpu"))
        torch.xpu.memory._set_allocator_settings('expandable_segments:False')

instantiate_device_type_tests(
    TestExpandableSegments, globals(), only_for="xpu", allow_xpu=True
)

# Owner(s): ["module: xpu"]
# run time xpu tests, but with the allocator using expandable segments
# now OneDNN does not support expandable segments, so below test is disabled

# if torch.xpu.is_available():
#     torch.xpu.memory._set_allocator_settings('expandable_segments:True')

# current_dir = os.path.dirname(os.path.abspath(__file__))

# current_file = os.path.basename(__file__)

# for fname in os.listdir(current_dir):
#     if fname.endswith('.py') and fname != current_file:
#         file_path = os.path.join(current_dir, fname)
#         print(f"Running {file_path} ...")
#         exec(compile(open(file_path, 'r').read(), file_path, mode='exec'))

if __name__ == "__main__":
    run_tests()
