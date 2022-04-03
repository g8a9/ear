import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

DEPENDENCIES = ["transformers"]

# This call to setup() does all the work
setup(
    name="ear",
    version="1.0.0",
    description="Entrop-based Attention Regularizationn",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/g8a9/ear",
    author="Giuseppe Attanasio",
    author_email="giuseppeattanasio6@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["ear"],
    include_package_data=True,
    install_requires=DEPENDENCIES,
)