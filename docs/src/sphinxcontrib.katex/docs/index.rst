.. include:: ../README.rst
    :end-line: 35

.. toctree::
    :hidden:
    :maxdepth: 2

    installation
    usage
    configuration
    macros
    examples
    contributing
    version-history


.. Add extra vertical space

|


.. code-block:: rst

    .. math::

        \text{Rendered with \KaTeX} \\[18pt]

        \gdef \f #1 {f(#1)}

        \f{x} = \int_{-\infty}^\infty
            \hat \f\xi\,e^{2 \pi i \xi x}
            \,d\xi

.. math::

    \text{Rendered with \KaTeX} \\[18pt]

    \gdef \f #1 {f(#1)}

    \f{x} = \int_{-\infty}^\infty
        \hat \f\xi\,e^{2 \pi i \xi x}
        \,d\xi
