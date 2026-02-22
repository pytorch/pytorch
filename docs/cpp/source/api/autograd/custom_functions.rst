Custom Autograd Functions
=========================

PyTorch allows you to define custom autograd functions with custom forward
and backward implementations.

Function Base Class
-------------------

.. doxygenstruct:: torch::autograd::Function
   :members:
   :undoc-members:

AutogradContext
---------------

.. doxygenstruct:: torch::autograd::AutogradContext
   :members:
   :undoc-members:

Creating Custom Functions
-------------------------

To create a custom autograd function, inherit from ``torch::autograd::Function``
and implement the static ``forward`` and ``backward`` methods:

**Example:**

.. code-block:: cpp

   class MyReLU : public torch::autograd::Function<MyReLU> {
    public:
     static torch::Tensor forward(
         torch::autograd::AutogradContext* ctx,
         torch::Tensor input) {
       ctx->save_for_backward({input});
       return input.clamp_min(0);
     }

     static torch::autograd::variable_list backward(
         torch::autograd::AutogradContext* ctx,
         torch::autograd::variable_list grad_outputs) {
       auto saved = ctx->get_saved_variables();
       auto input = saved[0];
       auto grad_output = grad_outputs[0];
       auto grad_input = grad_output * (input > 0).to(grad_output.dtype());
       return {grad_input};
     }
   };

   // Usage
   auto output = MyReLU::apply(input);

Custom Kernels and AutoDispatchBelowADInplaceOrView
---------------------------------------------------

For users implementing custom kernels who want to redispatch below ``Autograd`` dispatch
keys, use ``at::AutoDispatchBelowADInplaceOrView`` instead of ``InferenceMode``:

.. code-block:: cpp

   class ROIAlignFunction : public torch::autograd::Function<ROIAlignFunction> {
    public:
     static torch::autograd::variable_list forward(
         torch::autograd::AutogradContext* ctx,
         const torch::autograd::Variable& input,
         const torch::autograd::Variable& rois,
         double spatial_scale,
         int64_t pooled_height,
         int64_t pooled_width,
         int64_t sampling_ratio,
         bool aligned) {
       ctx->saved_data["spatial_scale"] = spatial_scale;
       ctx->saved_data["pooled_height"] = pooled_height;
       ctx->saved_data["pooled_width"] = pooled_width;
       ctx->saved_data["sampling_ratio"] = sampling_ratio;
       ctx->saved_data["aligned"] = aligned;
       ctx->saved_data["input_shape"] = input.sizes();
       ctx->save_for_backward({rois});

       at::AutoDispatchBelowADInplaceOrView guard;
       auto result = roi_align(
           input, rois, spatial_scale, pooled_height,
           pooled_width, sampling_ratio, aligned);
       return {result};
     }
   };

For customized inplace and view kernels, see the
`custom kernel tutorial <https://pytorch.org/tutorials/advanced/cpp_extension.html#backward-pass>`_
for more details.
