# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_export_persist_assert)
"""

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter


class TensorSubclassTests(torch._dynamo.test_case.TestCase):
    def test_tensor_property_on_tensor(self):
        def fn(x):
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = torch.randn([2, 2])
        x_.y = y_

        eager_result = fn(x_)

        graph = None

        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # We want to be very sure that this lifts y to inputs!
        self.assertEqual(placeholder_cnt, 2)

    def test_tensor_property_assigned_on_tensor(self):
        def fn(x, y):
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = torch.randn([2, 2])

        eager_result = fn(x_, y_)

        graph = None

        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_, y_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # y is already an input
        self.assertEqual(placeholder_cnt, 2)

    def test_const_property_on_tensor(self):
        def fn(x):
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = 4
        x_.y = y_

        eager_result = fn(x_)

        graph = None

        def grab_graph_backend(gm, inps):
            nonlocal graph
            graph = gm
            return gm

        fn = torch._dynamo.optimize(grab_graph_backend, nopython=True)(fn)
        compile_result = fn(x_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # We want to be very sure that this does not lifts y to inputs, as its a const
        self.assertEqual(placeholder_cnt, 1)

    def test_const_property_assigned_on_tensor(self):
        def fn(x, y):
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = 4

        eager_result = fn(x_, y_)

        fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        compile_result = fn(x_, y_)
        self.assertEqual(eager_result, compile_result)

    def test_guards_correctly_property_assigned_on_tensor_type_change(self):
        def fn(x, y):
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])

        fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        compile_result_const = fn(x_, 4)
        self.assertEqual(compile_result_const, x_ * 4)

        y = torch.randn([2, 2])
        compile_result_tensor = fn(x_, y)
        self.assertEqual(compile_result_tensor, x_ * y)

    def test_guards_correctly_property_assigned_on_tensor_type_change_inductor(self):
        def fn(x, y):
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])

        fn = torch._dynamo.optimize("inductor", nopython=True)(fn)
        compile_result_const = fn(x_, 4)
        self.assertEqual(compile_result_const, x_ * 4)

        y = torch.randn([2, 2])
        compile_result_tensor = fn(x_, y)
        self.assertEqual(compile_result_tensor, x_ * y)

    def test_complex_attr_access_with_graph_breaks(self):
        def fn(x, y, z):
            for t in x:
                t.y = y
                t.z = y * z

            print("Break")

            new_y = 1
            new_z = 1
            for t in x:
                new_y = t.y * new_y
                new_z = t.z * new_z

            return new_y, new_z

        x_0 = torch.randn([2, 2])
        x_1 = torch.randn([2, 2])
        x_2 = torch.randn([2, 2])
        x = [x_0, x_1, x_2]

        y = torch.randn([2, 2])
        z = 5

        eager_result = fn(x, y, z)

        counter = CompileCounter()
        fn = torch._dynamo.optimize(counter)(fn)

        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 9)
        # Graph for reference
        # <eval_with_key>.1 class GraphModule(torch.nn.Module):
        # def forward(self, L_x_0_dict_y_ : torch.Tensor, L_x_0_dict_z_ : torch.Tensor, L_x_1_dict_y_ :
        # torch.Tensor, L_x_1_dict_z_ : torch.Tensor, L_x_2_dict_y_ : torch.Tensor, L_x_2_dict_z_ : torch.Tensor):
        #     l_x_0_dict_y_ = L_x_0_dict_y_
        #     l_x_0_dict_z_ = L_x_0_dict_z_
        #     l_x_1_dict_y_ = L_x_1_dict_y_
        #     l_x_1_dict_z_ = L_x_1_dict_z_
        #     l_x_2_dict_y_ = L_x_2_dict_y_
        #     l_x_2_dict_z_ = L_x_2_dict_z_

        #     # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:158, code: new_y = t.y * new_y
        #     mul = l_x_0_dict_y_ * 1;  l_x_0_dict_y_ = None

        #     # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:159, code: new_z = t.z * new_z
        #     mul_1 = l_x_0_dict_z_ * 1;  l_x_0_dict_z_ = None

        #     # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:158, code: new_y = t.y * new_y
        #     mul_2 = l_x_1_dict_y_ * mul;  l_x_1_dict_y_ = mul = None

        #     # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:159, code: new_z = t.z * new_z
        #     mul_3 = l_x_1_dict_z_ * mul_1;  l_x_1_dict_z_ = mul_1 = None

        #     # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:158, code: new_y = t.y * new_y
        #     mul_4 = l_x_2_dict_y_ * mul_2;  l_x_2_dict_y_ = mul_2 = None

        #     # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:159, code: new_z = t.z * new_z
        #     mul_5 = l_x_2_dict_z_ * mul_3;  l_x_2_dict_z_ = mul_3 = None
        #     return (mul_4, mul_5)
