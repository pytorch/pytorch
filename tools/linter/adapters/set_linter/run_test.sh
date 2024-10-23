#!/bin/bash

set -ex

mypy *.py setlint/
ruff check --fix
ruff format
pytest -vvvv
