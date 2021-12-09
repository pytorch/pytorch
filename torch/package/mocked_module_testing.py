import torch
from io import BytesIO
from torch.package import PackageExporter, PackageImporter

class MockedModule:
    def __init__(self):
        self.repr = "I'm mocked"

class UsesMocked:
   def __init__(self, my_mocked_object):
    self.i_am_mocked = my_mocked_object

mocked = MockedModule()
has_mocked = UsesMocked(mocked)

buffer = BytesIO()
with PackageExporter(buffer) as e:
    e.mock(include = "MockedModule")
    e.intern("**")
    e.save_pickle("monke","more monke", has_mocked)
