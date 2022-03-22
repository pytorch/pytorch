import torch


# https://pytorch.org/docs/stable/jit_builtin_functions.html#builtin-functions


class BuiltinOpsModule(torch.nn.Module):
    def __init__(self):
        super(BuiltinOpsModule, self).__init__()

    def forward(self):
        # print("test")
        x = torch.tensor(1)
        y = torch.tensor(0.5)
        z = torch.randn(4)
        b = float(1)
        s = "abcde"
        l = ["1", "2", "test"]

        l[1] = "3"
        l.extend(["4"])
        l.pop(3)

        d = {"key": 1}
        d.clear()
        d.update({"key": 0})
        if "key" in d:
            d["key"] = 2

        d2 = {0: 100}
        if 0 in d2:
            d2.clear()
            d2[0] = 100
        return (
            # type
            bool(x),
            bool(x.item()),
            int(y),
            int(y.item()),
            int(l[0]),
            float(x),
            float(x.item()),
            float(l[0]),
            # math
            x & x,
            bool(x) & bool(x),
            int(x) & int(x),
            x | x,
            bool(x) | bool(x),
            int(x) | int(x),
            x << x,
            int(x) << int(x),
            x >> x,
            int(x) >> int(x),
            x ^ x,
            bool(x) ^ bool(x),
            int(x) ^ int(x),
            b * float(x),
            b * int(x),
            b + float(x),
            b - float(x),
            x.item() + y.item(),
            x.item() - y.item(),
            x.item() * y.item(),
            x.item() / y.item(),
            float(x) < float(y),
            float(x) <= float(y),
            float(x) > float(y),
            float(x) > int(y),
            float(x) >= float(y),
            float(x) >= int(y),
            float(x) == float(y),
            float(x) == int(y),
            float(x) != float(y),
            int(x) != float(y),
            float(x) / float(y),
            int(x) / int(y),
            max(x),
            max(x.item(), y.item()),
            max(int(x), int(y)),
            max(float(x), float(y)),
            min(x),
            min(x.item(), y.item()),
            min(int(x), int(y)),
            min(float(x), float(y)),
            # collection
            s[x],
            d["key"],
            d2[0],
            x.item() / y.item(),
            d.keys(),
            d.items(),
            d.values(),
            d2.values(),
            # string
            str(x),
            l[2].find("t"),
            l[2].replace("t", "x"),
            l[2].lower(),
            l[2].startswith("t"),
            l[2].split("t"),
            l[2].strip(),
            l[2].rstrip(),
            l[2].lstrip(),
            l[2][slice(2)],
            ord(l[2][0]),
            len(l[2]),
            len(s),
            len(z),
            len(d2),
        )
