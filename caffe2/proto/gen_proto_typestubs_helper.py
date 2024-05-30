import ast

with open("../python/__init__.py", "r") as f:
    tree = ast.parse(f.read())

print("\nDeviceType = int\n")
print("# These are freedom-patched into caffe2_pb2 in caffe2/proto/__init__.py")
for stmt in tree.body:
    if not isinstance(stmt, ast.Assign):
        continue
    target = stmt.targets[0]
    if not isinstance(target, ast.Attribute):
        continue
    if isinstance(target.value, ast.Name) and target.value.id == "caffe2_pb2":
        print(f"{target.attr}: int = DeviceType.PROTO_{target.attr}")
