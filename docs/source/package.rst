.. currentmodule:: torch.package

torch.package
=============

API Reference
-------------

.. autoclass:: torch.package.PackageExporter

  .. automethod:: __init__
  .. automethod:: close
  .. automethod:: deny
  .. automethod:: extern
  .. automethod:: file_structure
  .. automethod:: get_unique_id
  .. automethod:: mock
  .. automethod:: require_module
  .. automethod:: save_binary
  .. automethod:: save_extern_module
  .. automethod:: save_mock_module
  .. automethod:: save_module
  .. automethod:: save_pickle
  .. automethod:: save_source_file
  .. automethod:: save_source_string
  .. automethod:: save_text

.. autoclass:: torch.package.PackageImporter

  .. automethod:: __init__
  .. automethod:: file_structure
  .. automethod:: get_resource_reader
  .. automethod:: get_source
  .. automethod:: id
  .. automethod:: import_module
  .. automethod:: load_binary
  .. automethod:: load_text
