"""
Setup script for quantum_measurement_reduction package.
"""

import os

from setuptools import find_packages, setup


# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "A package for quantum measurement reduction techniques"


# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="quantum_measurement_reduction",
    version="0.1.0",
    author="Quantum Measurement Reduction Team",
    author_email="quantum@example.com",
    description="A comprehensive package for quantum measurement reduction techniques",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum_measurement_reduction",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
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
            "nbsphinx",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="quantum computing, measurement reduction, pauli operators, variance reduction",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/quantum_measurement_reduction/issues",
        "Source": "https://github.com/yourusername/quantum_measurement_reduction",
        "Documentation": "https://quantum-measurement-reduction.readthedocs.io/",
    },
)
