__import__("subpackage_1", globals(), fromlist=["PackageBSubpackage1Object_0"], level=1)
__import__("subpackage_0.subsubpackage_0", globals(), fromlist=[""], level=1)
__import__("subpackage_2", globals=globals(), locals=locals(), fromlist=["*"], level=1)

result = "package_b"
package_b_li = [123]

class PackageBObject:
    __slots__ = ["obj"]
    package_b_li = [789]

    def __init__(self, obj):
        self.obj = obj

    def return_result(self):
        return result
