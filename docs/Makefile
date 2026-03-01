# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?= -j auto -WT --keep-going
SPHINXBUILD   ?= sphinx-build
SPHINXPROJ    ?= PyTorch
SOURCEDIR     ?= source
BUILDDIR      ?= build
PYCMD         ?= python

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

figures:
	@$(PYCMD) source/scripts/build_activation_images.py
	@$(PYCMD) source/scripts/build_lr_scheduler_images.py

opset:
	@$(PYCMD) source/scripts/build_opsets.py

exportdb:
	@$(PYCMD) source/scripts/exportdb/generate_example_rst.py

docset: html
	doc2dash --name $(SPHINXPROJ) --icon $(SOURCEDIR)/_static/img/pytorch-logo-flame.png --enable-js --online-redirect-url https://pytorch.org/docs/ --force $(BUILDDIR)/html/

	# Manually fix because Zeal doesn't deal well with `icon.png`-only at 2x resolution.
	cp $(SPHINXPROJ).docset/icon.png $(SPHINXPROJ).docset/icon@2x.png
	convert $(SPHINXPROJ).docset/icon@2x.png -resize 16x16 $(SPHINXPROJ).docset/icon.png

html-stable:
	# stable differs from `make html` in two ways:
	# 1) The stable logo is used instead of the unstable logo
	# 2) There will not be a link to the stable docs.
	# See conf.py for more details.
	RELEASE=1 make html

.PHONY: help Makefile docset onnx exportdb serve linkcheck-local linkcheck-external

serve:
	@cd $(BUILDDIR)/html && $(PYCMD) -m http.server $(PORT)

linkcheck-local:
	@if [ ! -f "$(BUILDDIR)/html/index.html" ]; then \
		echo "ERROR: Built docs not found. Run 'make html' first."; \
		exit 1; \
	fi
	@echo "Running linkchecker on changed doc files..."
	@echo "Use './scripts/lint_doc_anchors.sh --all' for full site check."
	../scripts/lint_doc_anchors.sh HEAD~1 HEAD

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile figures onnx opset exportdb
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Removing everything under 'build' and 'source/generated'.."
	@rm -rf $(BUILDDIR)/html/ $(BUILDDIR)/doctrees $(SOURCEDIR)/generated $(BUILDDIR)/auto_gen_aten_op_list.csv
