torch.cfg
=========

.. automodule:: torch.cfg
.. currentmodule:: torch.cfg

``torch.cfg`` provides a small, typed control-flow graph IR for analyses and
transforms that need explicit basic blocks. Value names are globally unique
across a graph, even across blocks, so textual rendering and validation can use
them as stable identifiers.

Core IR
-------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Graph
    Block
    Value
    Instruction
    Successor
    Jump
    Branch
    Return

Value Specs
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Spec
    TensorSpec
    ScalarSpec
    TupleSpec
    ListSpec
    DictSpec
    OptionalSpec
    ObjectSpec

Utilities
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Literal
    literal
    Location
    ValidationError
    from_fx
