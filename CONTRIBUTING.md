## Contributing to PyTorch

If you are interested in contributing to PyTorch, your contributions will fall
into two categories:
1. You want to propose a new Feature and implement it
    - post about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/pytorch/pytorch/issues
    - Especially look at the Low Priority and Medium Priority issues
    - Pick an issue and comment on the task that you want to work on this feature
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bugfix, please send a Pull Request to
https://github.com/pytorch/pytorch

If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Developing locally with PyTorch

To locally develop with PyTorch, here are some tips:

1. Uninstall all existing pytorch installs
```
conda uninstall pytorch
pip uninstall torch
pip uninstall torch # run this command twice
```

2. Locally clone a copy of PyTorch from source:

```
git clone https://github.com/pytorch/pytorch
cd pytorch
```

3. Install PyTorch in `build develop` mode:

A full set of instructions on installing PyTorch from Source are here:
https://github.com/pytorch/pytorch#from-source

The change you have to make is to replace

`python setup.py install`

with

```
python setup.py build develop
```

This is especially useful if you are only changing Python files.

This mode will symlink the python files from the current local source tree into the
python install.

Hence, if you modify a python file, you do not need to reinstall pytorch again and again.

For example:
- Install local pytorch in `build develop` mode
- modify your python file torch/__init__.py (for example)
- test functionality
- modify your python file torch/__init__.py
- test functionality
- modify your python file torch/__init__.py
- test functionality

You do not need to repeatedly install after modifying python files.


Hope this helps, and thanks for considering to contribute.
