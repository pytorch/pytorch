functorch docs build
--------------------

## Build Locally

Install requirements:
```
pip install -r requirements.txt
```

One may also need to install [pandoc](https://pandoc.org/installing.html). On Linux we can use: `sudo apt-get install pandoc`. Or using `conda` we can use: `conda install -c conda-forge pandoc`.

To run the docs build:
```
make html
```

Check out the output files in `build/html`.

## Deploy

The functorch docs website does not updated automatically. We need to periodically regenerate it.

You need write permissions to functorch to do this. We use GitHub Pages to serve docs.

1. Build the docs
2. Save the build/html folder somewhere
3. Checkout the branch `gh-pages`.
4. Delete the contents of the branch and replace it with the build/html folder. `index.html` should be at the root.
5. Commit the changes and push the changes to the `gh-pages` branch.
