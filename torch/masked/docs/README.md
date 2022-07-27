# Github Pages for PyTorch MaskedTensor

---

gh-pages is the branch that holds the documentation for the main branch as well as the different releases. The main website is at https://pytorch.org/maskedtensor/

### Build instructions

To rebuild the documentation:

```
cd docs
pip install -r requirements.txt
make html
```

### Adding/syncing notebooks

If you're adding a notebook to the documentation and would like to sync the notebooks (see below), run:

```
jupytext --set-formats ipynb,md:myst source/notebooks/the_notebook.ipynb
```

To sync the ipynb and md notebooks, run:

```
jupytext --sync source/notebooks/*
```

### Updating documentation

Currently, the documentation is not updated automatically and is periodically generated using commands like:

```
# build the docs, which will end up in _build/html
# save them to a tmp intermediate directory
cd docs
pip install -r requirements.txt
jupytext --sync source/notebooks/*
make html
cp -r docs/_build/html/* /path/to/tmp_dir

# copy over the files to main
git checkout gh-pages
rm -rf main/*
cp -r /path/to/tmp_dir/* main

# push the files
git add main
git commit -m "generate new docs"
git push -u origin
```
