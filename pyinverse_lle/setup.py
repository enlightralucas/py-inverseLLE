"""
Setup script for PyInverseLLE package.
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyinverse_lle",
    version="0.1.0",
    author="Converted from MATLAB implementation",
    author_email="",
    description="Python implementation of genetic algorithm optimization for Kerr frequency comb generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyinverse-lle-optimize=pyinverse_lle.examples.optimize_flat_comb:main",
            "pyinverse-lle-analyze=pyinverse_lle.examples.analyze_results:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="photonics, frequency comb, genetic algorithm, optimization, lugiato-lefever",
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
)