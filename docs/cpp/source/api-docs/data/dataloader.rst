DataLoader
==========

The DataLoader batches samples from a dataset and optionally shuffles and
parallelizes the loading process. It is the main interface for iterating
over training data.

DataLoader Classes
------------------

.. doxygenclass:: torch::data::DataLoaderBase
   :members:
   :undoc-members:

DataLoaderOptions
-----------------

.. doxygenstruct:: torch::data::DataLoaderOptions
   :members:
   :undoc-members:

Creating a DataLoader
---------------------

Use ``make_data_loader`` to create a DataLoader from a dataset:

.. code-block:: cpp

   auto data_loader = torch::data::make_data_loader(
       std::move(dataset),
       torch::data::DataLoaderOptions()
           .batch_size(64)
           .workers(4));

   for (auto& batch : *data_loader) {
       auto data = batch.data;
       auto target = batch.target;
       // Train on batch
   }

Complete Training Example
-------------------------

.. code-block:: cpp

   #include <torch/torch.h>

   int main() {
     // Load dataset
     auto dataset = torch::data::datasets::MNIST("./data")
         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
         .map(torch::data::transforms::Stack<>());

     // Create data loader
     auto data_loader = torch::data::make_data_loader(
         std::move(dataset),
         torch::data::DataLoaderOptions().batch_size(64).workers(2));

     // Create model and optimizer
     auto model = std::make_shared<Net>();
     auto optimizer = torch::optim::Adam(model->parameters(), 0.001);

     // Training loop
     for (size_t epoch = 1; epoch <= 10; ++epoch) {
       for (auto& batch : *data_loader) {
         optimizer.zero_grad();
         auto output = model->forward(batch.data);
         auto loss = torch::nll_loss(output, batch.target);
         loss.backward();
         optimizer.step();
       }
     }
   }
