# PG3

![workflow](https://github.com/tomsilver/pg3/actions/workflows/pg3.yml/badge.svg)

Under development. This is not the official version of PG3: [this is](https://github.com/ryangpeixu/pg3).

## Requirements

- Python 3.8+
- Tested on MacOS Catalina

## Instructions For Contributing

### First Time

- (Highly recommended) Make a virtual environment:
  - `virtualenv venv`
  - `source venv/bin/activate`
- Clone this repository with submodules: `git clone https://github.com/tomsilver/pg3`
- Run `pip install -e .[develop]` to install the main dependencies for development.

### Developing

- You can't push directly to master. Make a new branch in this repository (don't use a fork, since that will not properly trigger the checks when you make a PR). When your code is ready for review, make a PR and request reviews from the appropriate people.
- To merge a PR, you need at least one approval, and you have to pass the 4 checks defined in `.github/workflows/pg3.yml`, which you can run locally as follows:
  - `pytest -s tests/ --cov-config=.coveragerc --cov=pg3/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered --durations=0`
  - `mypy .`
  - `pytest . --pylint -m pylint --pylint-rcfile=.pg3_pylintrc`
  - `./run_autoformat.sh`
- The first one is the unit testing check, which verifies that unit tests pass and that code is adequately covered. The "100" means that all lines in every file must be covered.
- The second one is the static typing check, which uses Mypy to verify type annotations.
- The third one is the linter check, which runs Pylint with the custom config file `.pg3_pylintrc` in the root of this repository. Feel free to edit this file as necessary.
- The fourth one is the autoformatting check, which uses the custom config files `.style.yapf` and `.isort.cfg` in the root of this repository.
