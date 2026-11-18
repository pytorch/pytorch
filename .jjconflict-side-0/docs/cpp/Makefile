# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    = -j auto
SPHINXBUILD   = python -m sphinx
SPHINXPROJ    = PyTorch
SOURCEDIR     = source
BUILDDIR      = build
PYCMD         = python
DOXYGEN       = doxygen

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean doxygen html

# Generate Doxygen XML (required for Breathe directives)
doxygen:
	@echo "Running Doxygen..."
	@cd $(SOURCEDIR) && $(DOXYGEN) Doxyfile

clean:
	@# Clean up sphinx build artifacts and Doxygen output.
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@rm -rf $(BUILDDIR)/xml

# Build HTML with Doxygen XML generation first
html: doxygen
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Check documentation coverage against curated allowlist and HTML formatting
coverage: html
	@echo "Running C++ docs coverage check..."
	@$(PYCMD) check_coverage.py

coverage-only:
	@echo "Running C++ docs coverage check (no rebuild)..."
	@$(PYCMD) check_coverage.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
