Version History
===============

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 0.5.1 (2019-08-13)
--------------------------

* Added: equation numbers in documentation (#16)
* Changed: subset of tests for sphinx<=1.6 (#23)
* Changed: several improvements to documentation


Version 0.5.0 (2019-07-25)
--------------------------

* Added: katex server side pre-rendering (#15)
* Changed: switch to Katex 0.10.2 (#17)
* Removed: deprecated Sphinx ``setup_math`` (#10)


Version 0.4.1 (2019-01-08)
--------------------------

* Fixed: macros example in documentation


Version 0.4.0 (2018-12-14)
--------------------------

* Added: Sphinx documentation and setup RTD page
* Added: Travis-CI tests
* Changed: KaTeX version 0.10.0
* Changed: make compatible with ``sphinx>=1.6``
* Removed: configuration option ``katex_version``


Version 0.3.1 (2018-10-08)
--------------------------

* Fixed: incompatibility with ``sphinx>=1.8`` (#8)


Version 0.3.0 (2018-09-06)
--------------------------

* Added: allow for user defined autorendering delimiters (#7)
* Fixed: bug if ``katex_options`` was blank (#5)


Version 0.2.0 (2018-06-22)
--------------------------

* Added: document all configuration settings
* Added: automatic setting of delimiters for KaTeX auto-renderer
* Removed: ``katex_macros`` option


Version 0.1.6 (2018-04-12)
--------------------------

* Added: equation numbering across pages with ``sphinx>=1.7``
* Changed: KaTeX version 0.9.0


Version 0.1.5 (2017-12-19)
--------------------------

* Added: helper function to convert LaTeX defs to KaTeX macros
* Changed: improvement of code readability
* Fixed: mouse over for equation numbers in Firefox


Version 0.1.4 (2017-11-27)
--------------------------

* Changed: move equation numbers to the right and center vertically


Version 0.1 (2017-11-24)
------------------------

* Added: initial release


.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
