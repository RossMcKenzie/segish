import re

import setuptools

VERSIONFILE="segish/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Segish",
    version=verstr,
    author="Ross McKenzie",
    author_email="rmwmckenzie@gmail.com",
    description="Segmentation using optimisation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RossMcKenzie/segish",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "imageio",
        "scikit-learn"
    ]
)
