#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Installation:
    pip install git+https://github.com/Erotemic/graphid.git

Developing:
    git clone https://github.com/Erotemic/graphid.git
    pip install -e graphid

Pypi:
     # Presetup
     pip install twine

     # First tag the source-code
     VERSION=$(python -c "import setup; print(setup.version)")
     echo $VERSION
     git tag $VERSION -m "tarball tag $VERSION"
     git push --tags origin master

     # NEW API TO UPLOAD TO PYPI
     # https://packaging.python.org/tutorials/distributing-packages/

     # Build wheel or source distribution
     python setup.py bdist_wheel --universal

     # Use twine to upload. This will prompt for username and password
     # If you get an error:
     #   403 Client Error: Invalid or non-existent authentication information.
     # simply try typing your password slower.
     twine upload --username erotemic --skip-existing dist/*

     # Check the url to make sure everything worked
     https://pypi.org/project/graphid/

     # ---------- OLD ----------------
     # Check the url to make sure everything worked
     https://pypi.python.org/pypi?:action=display&name=graphid

"""
from setuptools import setup


def parse_version(package):
    """
    Statically parse the version number from __init__.py

    CommandLine:
        python -c "import setup; print(setup.parse_version('graphid'))"
    """
    from os.path import dirname, join
    import ast
    init_fpath = join(dirname(__file__), package, '__init__.py')
    with open(init_fpath) as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if target.id == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.md')
    if exists(readme_fpath):
        textlines = []
        with open(readme_fpath, 'r') as f:
            for line in f.readlines():
                textlines += [line]
        text = ''.join(textlines).strip()
        return text
    return ''


def parse_requirements(fname='requirements.txt'):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    from os.path import dirname, join, exists
    import re
    require_fpath = join(dirname(__file__), fname)
    # This breaks on pip install, so check that it exists.
    if exists(require_fpath):
        with open(require_fpath, 'r') as f:
            packages = []
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.startswith('-e '):
                        package = line.split('#egg=')[1]
                        packages.append(package)
                    else:
                        pat = '|'.join(['>', '>=', '=='])
                        package = re.split(pat, line)[0]
                        packages.append(package)
            return packages
    return []


version = parse_version('graphid')  # needs to be a global var for git tags

if __name__ == '__main__':
    install_requires = parse_requirements('requirements.txt')

    setup(
        name='graphid',
        version=version,
        author='Jon Crall',
        description='The graph identification algorithm from my thesis',
        long_description=parse_description(),
        install_requires=install_requires,
        author_email='erotemic@gmail.com',
        url='https://github.com/Erotemic/graphid',
        license='Apache 2',
        packages=['graphid'],
        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 3 - Alpha',
            'Topic :: Software Development :: Libraries :: Python Modules',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 3',
        ],
    )
