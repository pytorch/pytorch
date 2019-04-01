import os
import re
import yaml
import unittest


path = os.path.dirname(os.path.realpath(__file__))
aten_native_yaml = os.path.join(path, '../aten/src/ATen/native/native_functions.yaml')
whitelist = [
    'max', 'min', 'median', 'mode', 'kthvalue', 'svd', 'symeig', 'eig',
    'pstrf', 'qr', 'geqrf', 'solve', 'slogdet', 'sort', 'topk',
]


class TestNamedTupleAPI(unittest.TestCase):

    def test_field_name(self):
        regex = re.compile(r"^(\w*)\(")
        file = open(aten_native_yaml, 'r')
        for f in yaml.load(file.read()):
            f = f['func']
            ret = f.split('->')[1].strip()
            name = regex.findall(f)[0]
            if name in whitelist or name.endswith('_backward') or \
               name.endswith('_forward'):
                continue
            if not ret.startswith('('):
                continue
            ret = ret[1:-1].split(',')
            for r in ret:
                r = r.strip()
                self.assertEqual(len(r.split()), 1,
                                 'only whitelisted operators are allowed to have named return type, got ' + name)
        file.close()


if __name__ == '__main__':
    unittest.main()
