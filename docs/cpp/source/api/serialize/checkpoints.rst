Checkpoints
===========

Checkpoints save the complete training state so you can resume training
after interruption. A checkpoint typically includes:

- Model parameters
- Optimizer state (momentum buffers, learning rates)
- Current epoch number
- Best validation loss/accuracy

Creating Checkpoints
--------------------

.. code-block:: cpp

   void save_checkpoint(
       std::shared_ptr<Net> model,
       torch::optim::Adam& optimizer,
       int epoch,
       const std::string& path) {
     torch::serialize::OutputArchive archive;
     model->save(archive);
     archive.write("epoch", torch::tensor(epoch));
     optimizer.save(archive);
     archive.save_to(path);
   }

Loading Checkpoints
-------------------

.. code-block:: cpp

   int load_checkpoint(
       std::shared_ptr<Net> model,
       torch::optim::Adam& optimizer,
       const std::string& path) {
     torch::serialize::InputArchive archive;
     archive.load_from(path);
     model->load(archive);
     torch::Tensor epoch_tensor;
     archive.read("epoch", epoch_tensor);
     optimizer.load(archive);
     return epoch_tensor.item<int>();
   }

Complete Checkpoint Example
---------------------------

.. code-block:: cpp

   #include <torch/torch.h>
   #include <iostream>
   #include <filesystem>

   struct Net : torch::nn::Module {
     Net() {
       fc1 = register_module("fc1", torch::nn::Linear(784, 256));
       fc2 = register_module("fc2", torch::nn::Linear(256, 10));
     }

     torch::Tensor forward(torch::Tensor x) {
       x = torch::relu(fc1->forward(x.view({-1, 784})));
       return fc2->forward(x);
     }

     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
   };

   int main() {
     auto model = std::make_shared<Net>();
     auto optimizer = torch::optim::Adam(model->parameters(), 1e-3);

     int start_epoch = 0;
     const std::string checkpoint_path = "checkpoint.pt";

     // Resume from checkpoint if it exists
     if (std::filesystem::exists(checkpoint_path)) {
       std::cout << "Loading checkpoint..." << std::endl;
       start_epoch = load_checkpoint(model, optimizer, checkpoint_path);
       std::cout << "Resuming from epoch " << start_epoch << std::endl;
     }

     // Training loop
     for (int epoch = start_epoch; epoch < 100; ++epoch) {
       // ... training code ...

       // Save checkpoint every 10 epochs
       if ((epoch + 1) % 10 == 0) {
         save_checkpoint(model, optimizer, epoch + 1, checkpoint_path);
         std::cout << "Saved checkpoint at epoch " << epoch + 1 << std::endl;
       }
     }

     return 0;
   }

Best Practices
--------------

1. **Save periodically**: Save checkpoints at regular intervals (e.g., every epoch
   or every N batches) to minimize lost work.

2. **Keep multiple checkpoints**: Maintain the last few checkpoints in case the
   most recent one is corrupted or represents a poor model state.

3. **Include all state**: Save everything needed to resume training, including
   learning rate scheduler state if using one.

4. **Verify checkpoints**: Occasionally verify that checkpoints can be loaded
   correctly.
