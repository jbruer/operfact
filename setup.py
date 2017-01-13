from setuptools import setup

setup(
    name='operfact',
    version='0.1.0',
    author='John J. Bruer',
    author_email='jbruer@cms.caltech.edu',
    packages=['operfact'],
    package_dir={'operfact': 'operfact'},
    url='http://github.com/jbruer/operfact',
    license='All rights reserved',
    zip_safe=False,
    description='A tool to model structured low-rank operator problems with cvxpy.',
    install_requires=["cvxpy",
                      "numpy >= 1.9",
                      "scipy >= 0.15"],
)
