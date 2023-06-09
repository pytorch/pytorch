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
        
        # We want to be very sure that this does not lift y to inputs!
        self.assertEqual(placeholder_cnt, 1)


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
        
        # We want to be very sure that this does not lift x.y to inputs!
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
        
        # We want to be very sure that this does not lift y to inputs!
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

        y =  torch.randn([2, 2])
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

        y =  torch.randn([2, 2])
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
        self.assertEqual(counter.op_count, 21)
        # Graph for reference
        # def forward(self, L_x_0_ : torch.Tensor, L_x_1_ : torch.Tensor, L_x_2_ : torch.Tensor):
        # l_x_0_ = L_x_0_
        # l_x_1_ = L_x_1_
        # l_x_2_ = L_x_2_
        
        # # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:153, code: print("Break")
        # getattr_1 = l_x_0_.y
        # getattr_2 = l_x_0_.z;  l_x_0_ = None
        # getattr_3 = l_x_1_.y
        # getattr_4 = l_x_1_.z;  l_x_1_ = None
        # getattr_5 = l_x_2_.y
        # getattr_6 = l_x_2_.z;  l_x_2_ = None
        
        # # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:158, code: new_y = t.y * new_y
        # mul = getattr_1 * 1;  getattr_1 = None
        
        # # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:159, code: new_z = t.z * new_z
        # mul_1 = getattr_2 * 1;  getattr_2 = None
        
        # # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:158, code: new_y = t.y * new_y
        # mul_2 = getattr_3 * mul;  getattr_3 = mul = None
        
        # # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:159, code: new_z = t.z * new_z
        # mul_3 = getattr_4 * mul_1;  getattr_4 = mul_1 = None
        
        # # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:158, code: new_y = t.y * new_y
        # mul_4 = getattr_5 * mul_2;  getattr_5 = mul_2 = None
        
        # # File: /scratch/voz/work/pytorch/test/dynamo/test_tensor_subclass.py:159, code: new_z = t.z * new_z
        # mul_5 = getattr_6 * mul_3;  getattr_6 = mul_3 = None
        # return (mul_4, mul_5)

    def test_tensor_subclass_with_attr(self):
        class TensorTestSubclass(torch.Tensor):
            associated_tensors = []

            def _associate_tensor(self, x):
                self.associated_tensors.append(x)
            
            def _last_associated(self):
                return self.associated_tensors[-1]

        def fn(sub: TensorTestSubclass, x, y, z):
            sub._associate_tensor(x)
            sub._associate_tensor(y)
            sub._associate_tensor(z)
        
            return sub._last_associated() * sub
        
        tts = TensorTestSubclass([[0.5, 0.5], [0.5, 0.5]])
        x = torch.randn([2, 2])
        y = torch.randn([2, 2])
        z = torch.randn([2, 2])
        eager_result = fn(tts, x, y, z)
        fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        compile_result = fn(tts, x, y, z)
        self.assertEqual(compile_result, eager_result)

    
    