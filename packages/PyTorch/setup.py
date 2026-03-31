from setuptools import find_packages, setup

exec(open('pytorch_mr/__version__.py').read())

long_description = """
# Modern Robotics (PyTorch)

PyTorch implementation of the mathematical routines from
_Modern Robotics: Mechanics, Planning, and Control_.

This package mirrors the reference NumPy `modern_robotics` library with batched tensor support.

Install the NumPy reference separately from `packages/Python` if you need parity tests or side-by-side use:

`pip install -e ../Python` (from this directory's parent) or `pip install modern_robotics`.
"""

setup(
    name="pytorch_mr",
    version=__version__,
    author="Huan Weng, Mikhail Todes, Jarvis Schultz, Bill Hunt, Ayush Gaggar",
    author_email="huanweng@u.northwestern.edu",
    description=(
        "Modern Robotics: Mechanics, Planning, and Control — PyTorch implementation"
    ),
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="kinematics robotics dynamics pytorch",
    url="http://modernrobotics.org/",
    packages=find_packages(include=["pytorch_mr", "pytorch_mr.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "numpy",
        "torch",
    ],
    platforms="Linux, Mac, Windows",
)
