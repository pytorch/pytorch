Datasets
========

The dataset abstraction defines how to access individual samples in your data.
All datasets inherit from ``Dataset`` and must implement ``get()`` and ``size()``.

Dataset Base Class
------------------

.. doxygenclass:: torch::data::datasets::Dataset
   :members:
   :undoc-members:

.. doxygenclass:: torch::data::datasets::BatchDataset
   :members:
   :undoc-members:

Custom Dataset Example
----------------------

.. code-block:: cpp

   class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
    public:
     explicit CustomDataset(const std::string& root) {
       // Load data from root directory
     }

     torch::data::Example<> get(size_t index) override {
       return {images_[index], labels_[index]};
     }

     torch::optional<size_t> size() const override {
       return images_.size(0);
     }

    private:
     torch::Tensor images_, labels_;
   };

Built-in Datasets
-----------------

MNIST
^^^^^

.. doxygenclass:: torch::data::datasets::MNIST
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto dataset = torch::data::datasets::MNIST("./data")
       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
       .map(torch::data::transforms::Stack<>());

Example Struct
--------------

.. doxygenstruct:: torch::data::Example
   :members:
   :undoc-members:
