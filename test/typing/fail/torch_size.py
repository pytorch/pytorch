from torch import Size


s1 = Size([1, 2, 3])
s1 + ("foo",)  # E: Unsupported operand types
