Development notes
=================

### One-time setup
You probably want to install the toolchain:

    $ npm install

### Auto-generate files
This will (re-) build auto-generated files (stylus, etc).

    $ make
    $ make -B       # Recompiles things

### Update vendor files
This will download files.

    $ cd support/vendor/; make *.js

### Make a release
This will concat/compress things into a build in `/v/0.8.0`.

    $ make v/0.8.0

To do
-----

- Sectioning
