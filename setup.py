"""Setup script.

Usage examples:

    pip install -e .
    pip install -e .[develop]
"""
from setuptools import find_packages, setup

setup(name="pg3",
      version="0.0.0",
      packages=find_packages(include=["pg3", "pg3.*"]),
      install_requires=[
          "graphlib-backport",
          "numpy",
          "pathos",
          "pyperplan",
          "typing-extensions",
      ],
      package_data={"pg3": ["py.typed"]},
      extras_require={
          "develop": [
              "mypy", "pytest-cov>=2.12.1", "pytest-pylint>=0.18.0",
              "yapf==0.32.0", "docformatter", "isort"
          ]
      })
