from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='hybrid4cast',
    version='1.0.0',
    description='ts forecast using cnn models using ES and CNN',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='Microsoft',
    author_email='toguan@microsoft.com',
    license='BSD',
    packages=find_packages(),
    install_requires=[],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
