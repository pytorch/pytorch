Inductor CPU backend debugging and profiling
==============================================

**Author**: `Liao Xuan <https://github.com/Valentine233>`_, `Zhu Haozhe <https://github.com/zhuhaozhe>`_

This tutorial is intended to introduce the usage, debugging and performance profiling for ``torch.compile`` with Inductor CPU backend.

Usage
--------------

Start with an example
^^^^^^^^^^^^^^^^^^^

Here is a simple example to run the ``torch.compile`` with Inductor.

.. code-block:: python

	import torch
	
	def fn(x):
	    return torch.neg(x)
	
	x = torch.randn((2, 4, 28))
	compiled_fn = torch.compile(fn) # backend=inductor as default
	result = compiled_fn(x)

Get more loggings
^^^^^^^^^^^^^^^^^^^

However, the above code would not give any debugging info. If we want to get more useful logging, one way is to add an environment variable.

.. code:: shell

	TORCH_COMPILE_DEBUG=1 python xx.py

The time taken in each step is shown. This also does the graph visualization and prints the output code. In logging, a temperate debug tracing directory like this can be found.

.. code:: shell

	torch._inductor.debug: [WARNING] model___20 debug trace: /tmp/torchinductor_root/rx/crxfi2ybd7yp5sbj2pnhw33wfhtdw7wumvrobyp5sjvdui5ktjc2.debug

The directory saves several files for debugging.

+-------------------------+----------------------------------------------------------+
| fx_graph_readable.py    | Readable FX graph, post decomps                          |
+-------------------------+----------------------------------------------------------+
| fx_graph_runnable.py    | Executable FX graph, post decomps, pre pattern match     |
+-------------------------+----------------------------------------------------------+
| fx_graph_transformed.py | Transformed FX graph, post pattern match                 |
+-------------------------+----------------------------------------------------------+
| ir_post_fusion.txt      | Inductor IR before fusion                                |
+-------------------------+----------------------------------------------------------+
| ir_pre_fusion.txt       | Inductor IR after fusion                                 |
+-------------------------+----------------------------------------------------------+
| output_code.py          | Generated Python code for graph, with cpp/triton kernels |
+-------------------------+----------------------------------------------------------+


``fx_graph_runnable.py`` and ``output_code.py`` are both runnable and editable in order to make debugging easier.


Here is another way to print logging for Inductor.

.. code:: shell

	TORCH_LOGS="+inductor,output_code,schedule" python xx.py

+--------------+-------------------------------------------------------------+
| +inductor    | Set the logging level of Inductor to DEBUG, default is INFO |
+--------------+-------------------------------------------------------------+
| +output_code | Print output code with cpp/triton kernels                   |
+--------------+-------------------------------------------------------------+
| +schedule    | Print reasons for not doing vectorization in cpp kernels    |
+--------------+-------------------------------------------------------------+

Configs to do deeper analysis
^^^^^^^^^^^^^^^^^^^

Moreover, there are several config parameters helping the analysis.

+--------------------------------------------------+---------------------------------------------------------------------+
| torch._inductor.config.max_fusion_size           | Set the maximum number of nodes allowed in one fusion               |
+--------------------------------------------------+---------------------------------------------------------------------+
| torch._inductor.config.cpp.simdlen               | Specify the bit width for cpp vectorization                         |
+--------------------------------------------------+---------------------------------------------------------------------+
| torch._inductor.config.cpp.min_chunk_size        | Set the minimum number of workloads one thread should at least take |
+--------------------------------------------------+---------------------------------------------------------------------+
| torch._inductor.config.cpp.enable_kernel_profile | Allow cpp kernel performance profiling via profiler                 |
+--------------------------------------------------+---------------------------------------------------------------------+


Debugging
--------------

Determine component of error
^^^^^^^^^^^^^^^^^^^

When encountering errors or accuracy problem, a straightforward solution to find the bug is to narrow down the problem. The first thing to do is to determine the component where error occurs. Luckily, it can be simply achieved by changing the backend of ``torch.compile``.

+----------------------------------------+-----------------------------------------+
| torch.compile(fn, backend="eager")     | Enable Dynamo                           |
+----------------------------------------+-----------------------------------------+
| torch.compile(fn, backend="aot_eager") | Enable Dynamo + AOT autograd            |
+----------------------------------------+-----------------------------------------+
| torch.compile(fn, backend="inductor")  | Enable Dynamo + AOT autograd + Inductor |
+----------------------------------------+-----------------------------------------+

If the model can successfully run when backend is eager or aot_eager while it fails with inductor, we can narrow down the failure to Inductor.


Example
^^^^^^^^^^^^^^^^^^^

Here is an example for the subsequent debugging.

.. code-block:: python

	import torch
	from torch._dynamo.utils import same
	
	def foo(x1, x2):
	    a = torch.neg(x1)
	    b = torch.maximum(x2, a)
	    y = torch.cat([b], dim=0)
	    return y
	
	x1 = torch.randint(256, (1,), dtype=torch.uint8)
	x2 = torch.randint(256, (8390,), dtype=torch.uint8)
	
	expected_result = fn(x1, x2)
	
	compiled_fn = torch.compile(fn)
	actual_result = compiled_fn(x1, x2)
	
	assert same(expected_result, actual_result) == True


The implementation of ``neg`` in cpp codegen is as follows.

.. code-block:: python

	def neg(x):
	        return f"decltype({x})(-{x})"


In order to demonstrate the debugging, we will modify the function to a wrong one later.

Errors debugging
^^^^^^^^^^^^^^^^^^^

If it occurs a compile error, the root cause is usually shown in traceback log.

For example, the ``neg`` function is modified like this.

.. code-block:: python

	def neg(x):
	        return f"-{x}"


The logging gives the following compile error with a rather clear reason. In this case, the root cause is that data types of maximum's inputs are inconsistent.

.. code:: shell

	…
	torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
	CppCompileError: C++ compile error
	…
	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp: In function ‘void kernel(const unsigned char*, const unsigned char*, unsigned char*)’:
	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:14:53: error: no matching function for call to ‘max_propagate_nan(unsigned char&, int&)’
	   14 |             auto tmp3 = max_propagate_nan(tmp0, tmp2);
	      |                                                     ^
	In file included from /tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:2:
	/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note: candidate: ‘template<class scalar_t> scalar_t max_propagate_nan(scalar_t, scalar_t)’
	   27 | inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
	      |                 ^~~~~~~~~~~~~~~~~
	/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note:   template argument deduction/substitution failed:
	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:14:53: note:   deduced conflicting types for parameter ‘scalar_t’ (‘unsigned char’ and ‘int’)
	   14 |             auto tmp3 = max_propagate_nan(tmp0, tmp2);
	      |                                                     ^


Otherwise, if the model runs with other errors, we can do the model code reduction until finding the minimum code snippet with failure. Thus, the target operators and kernels are located.


Accuracy debugging
^^^^^^^^^^^^^^^^^^^

The accuracy problem refers the case where outputs of backends eager and inductor are different. As FX graph is generated before Inductor and output code is generated after Inductor, we can narrow down the problem by comparing their outputs.

If a model has several graphs, the first step is to compare the final outputs of FX graph and output code for each graph, given the same input. The target is to find the first graph occurring error or with different outputs. Binary search is suggested to use for efficiency.

When a model has only one graph or the problematic graph has been found with the above step, compare the intermediate outputs of FX graph and output code in each graph, given the same input. The idea is to continuously narrow down the problem.

For example, we modify the ``neg`` function like this.

.. code-block:: python

	def neg(x):
	        return f"decltype({x})(2 * {x})"


An accuracy problem would be raised as follows.

.. code:: shell

	torch._dynamo.utils: [ERROR] Accuracy failed: allclose not within tol=0.0001
	Traceback (most recent call last):
	  File "test_script.py", line 18, in <module>
	    assert same(expected_result, actual_result) == True
	AssertionError


By comparing the intermediate outputs of FX graph and output code, it would be found that outputs are already different after doing ``torch.neg``.

Specifically, the modifications of FX graph and output code are attached.

*Change of FX graph*

.. code-block:: python

	# Before
	class Repro(torch.nn.Module):
	    def __init__(self):
	        super().__init__()
	
	    def forward(self, arg0_1, arg1_1):
	        neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
	        maximum = torch.ops.aten.maximum.default(arg1_1, neg);  arg1_1 = neg = None
	        clone = torch.ops.aten.clone.default(maximum);  maximum = None
	        return (clone,)
	
	# After
	class Repro(torch.nn.Module):
	    def __init__(self):
	        super().__init__()
	    
	    def forward(self, arg0_1, arg1_1):
	        neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
	        return (neg,)


*Change of output code*

.. code-block:: python

	# Before
	cpp_fused_cat_maximum_neg_0 = async_compile.cpp('''
	#include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
	extern "C" void kernel(const long* in_ptr0,
	                       const long* in_ptr1,
	                       long* out_ptr0)
	{
	    {
	        #pragma GCC ivdep
	        for(long i0=static_cast<long>(0L); i0<static_cast<long>(8390L); i0+=static_cast<long>(1L))
	        {
	            auto tmp0 = in_ptr0[static_cast<long>(i0)];
	            auto tmp1 = in_ptr1[static_cast<long>(0L)];
	            auto tmp2 = decltype(tmp1)(2 * tmp1);
	            auto tmp3 = max_propagate_nan(tmp0, tmp2);
	            out_ptr0[static_cast<long>(i0)] = tmp3;
	        }
	    }
	}
	''')
	
	def call(args):
	    arg0_1, arg1_1 = args
	    args.clear()
	    buf0 = empty_strided((8390, ), (1, ), device='cpu', dtype=torch.int64)
	    cpp_fused_cat_maximum_neg_0(c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
	    del arg0_1
	    del arg1_1
	    return (buf0, )
	
	# After
	cpp_fused_cat_maximum_neg_0 = async_compile.cpp('''
	#include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
	extern "C" void kernel(const long* in_ptr0,
	                       const long* in_ptr1,
	                       long* out_ptr0)
	{
	    {
	        auto tmp1 = in_ptr1[static_cast<long>(0L)];
	        auto tmp2 = decltype(tmp1)(2 * tmp1);
	        out_ptr0[static_cast<long>(0L)] = tmp2;
	    }
	}
	''')
	
	def call(args):
	    arg0_1, arg1_1 = args
	    args.clear()
	    buf0 = empty_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
	    cpp_fused_cat_maximum_neg_0(c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
	    del arg0_1
	    del arg1_1
	    return (buf0, )


Note that there exists a debugging tool provided by PyTorch, called `Minifier <https://pytorch.org/docs/stable/dynamo/troubleshooting.html>`_. It helps us automatically generate a minified problematic graph.


Performance profiling
--------------
TODO: Haozhe


Future work
--------------

Implement and up-stream the debug tools
	1. **Cosim**: Merge graphs of a model into a single large graph. Thus, graphs can be compared quickly between different versions of PyTorch. `#102958 <https://github.com/pytorch/pytorch/pull/102958>`_
	2. **Graph matching**: In order to know what each kernel does, this tool matches cpp kernel with FX graph operators and adds corresponding operators before each kernel in cpp output code. `#102958 <https://github.com/pytorch/pytorch/pull/102958>`_
	3. **Save inputs and outputs**: For the purpose of reproducing rapidly the failure of a large model, it is necessary to add serializations for the inputs and outputs among graphs and intermediate outputs in graphs.
	4. **Test case generation**: When a user has found the operators which are inefficient with cpp kernels, a tool is needed to automatically write a test case. Specifically, one test case can be generated for each kernel, with the corresponding small FX graph and input.
	5. **Minifier optimization**: Keep refining Minifier and make it adapted for more scenarios.
