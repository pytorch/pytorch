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
import package_b
obj = package_b.subpackage_1.PackageBSubpackage1Object_0
with PackageExporter(buffer) as e:
    e.mock(include = "package_b.subpackage_1.PackageBSubpackage1Object_0")
    e.intern("**")
    e.save_pickle("monke","more monke", obj)
