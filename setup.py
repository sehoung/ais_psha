from setuptools import setup, find_packages

setup(
    name="AIS_PSHA",  # Unique name on PyPI
    version="1.0.0",
    author="SoungEil Houng",
    author_email="shoung@berkeley.edu",
    description="Adaptive Importance Sampling PSHA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sehoung/ais_psha",
    packages=find_packages(),  # Automatically finds all sub-packages
    install_requires=[
        "numpy",  # Add required dependencies
    ],
    classifiers=[],
    #python_requires=">=3.6",
)
