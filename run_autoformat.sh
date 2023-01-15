#!/bin/bash
yapf -i -r --style .style.yapf --exclude '**/third_party' pg3
yapf -i -r --style .style.yapf tests
yapf -i -r --style .style.yapf *.py
docformatter -i -r . --exclude venv pg3/third_party
isort .
