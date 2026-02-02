Archives
========

Archives provide a lower-level interface for serialization, allowing you to
save multiple values to a single file with named keys.

OutputArchive
-------------

.. doxygenclass:: torch::serialize::OutputArchive
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   torch::serialize::OutputArchive archive;
   archive.write("tensor1", tensor1);
   archive.write("tensor2", tensor2);
   archive.save_to("multi_tensor.pt");

InputArchive
------------

.. doxygenclass:: torch::serialize::InputArchive
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   torch::serialize::InputArchive archive;
   archive.load_from("multi_tensor.pt");

   torch::Tensor tensor1, tensor2;
   archive.read("tensor1", tensor1);
   archive.read("tensor2", tensor2);

Saving Multiple Values
----------------------

Archives are useful when you need to save multiple related values together:

.. code-block:: cpp

   // Save multiple tensors and metadata
   torch::serialize::OutputArchive out_archive;
   out_archive.write("weights", model_weights);
   out_archive.write("biases", model_biases);
   out_archive.write("epoch", torch::tensor(current_epoch));
   out_archive.write("loss", torch::tensor(best_loss));
   out_archive.save_to("training_state.pt");

   // Load them back
   torch::serialize::InputArchive in_archive;
   in_archive.load_from("training_state.pt");

   torch::Tensor weights, biases, epoch_tensor, loss_tensor;
   in_archive.read("weights", weights);
   in_archive.read("biases", biases);
   in_archive.read("epoch", epoch_tensor);
   in_archive.read("loss", loss_tensor);

   int epoch = epoch_tensor.item<int>();
   float loss = loss_tensor.item<float>();
