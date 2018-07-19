# Doxygen Notes

Doxygen seems to behave better if the processing between C++ and Python was split up. This is why there are two different links to cover each API.

C++ API docs work out of the box with the Caffe2 source code. Python docs require “python blocks (http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html#pythonblocks)” which are (often) currently missing in the Python code.

The Python script called “process.py” that resides in the /docs folder is to prepare the docs by looking for the block and if it doesn't exist prepend and customize the python blocks section with the module's path (e.g. Module caffe2.python.examples.char_rnn). It was noted that you need to delete the previous version of docs when you regenerate the docs or else things get messy, so the script deals with that as well.

The doxygen customization includes these files in the doxygen folder:

* header.html - logo links back to the main docs page
* footer.html - includes the Facebook OSS footer
* stylesheet.css - doxygen's default CSS; tweaked to fix formatting problems with the custom logo, header, and footer
* main.css - copied from the caffe2ai CSS, so this should be refreshed after the design changes (this overrides/extends stylesheet.css)

It also extracts info from markdown files found in the source tree. A legacy installation file was in the /docs folder and this was removed. These file show up in the top navigation under “Related Pages”.

The flow to create the API documents is simple now:

1. Run /caffe2_root/docs/process.py
2. (TODO automatically) Copy the doxygen-c and doxygen-python folders created by the script to the gh-pages branch.

Settings that were customized:

OPTIMIZE_OUTPUT_JAVA - turned on for Python config, off for C++ config
USE_MDFILE_AS_MAINPAGE  - use to flag a markdown file for the mainpage
EXTRACT_ALL
QUIET
WARN_IF_UNDOCUMENTED
FILE_PATTERNS
DOT_MULTI_TARGETS = YES
JAVADOC_AUTOBRIEF = YES
QUIET = YES
SOURCE_BROWSER = YES
VERBATIM_HEADERS = NO
SHOW_NAMESPACES = NO for C++ config

Not using this (was in old config file, but seems to be for Latex):
EXTRA_PACKAGES = amsmath \
amsfonts \
xr

### NOTE / TODO:

useful for xcode, currently off
GENERATE_DOCSET = NO

Look at search engine integration, xml output, etc
EXTERNAL_SEARCH = YES

### Other Notes

To achieve better output in the Python docs:
http://stackoverflow.com/questions/7690220/how-to-document-python-function-parameter-types

Swap this kind of formatting into py files:

```
def my_method(x, y):"""!
    my_method description

    @type x: int
    @param x: An integer

    @type y: int|string
    @param y: An integer or string

    @rtype: string
    @return: Returns a sentence with your variables in it
    """return "Hello World! %s, %s" % (x,y)
```

Note that the bang (!) is added after the opening comment """! - this seems to do the trick and the remaining comments will be nicely parsed by Doxygen.

