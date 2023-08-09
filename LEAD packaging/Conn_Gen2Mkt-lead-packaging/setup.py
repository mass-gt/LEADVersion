"""A setuptools based setup module for the parcel generation 2 parcel market connector.
"""

import pathlib
from os.path import join
from ast import literal_eval

from io import open

# Always prefer setuptools over distutils
from setuptools import setup, find_packages


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Get the version from the __init__.py file
def get_version():
    """Scan __init__ file for __version__ and retrieve."""

    finit = join(here, 'src', 'gen2mkt', '__init__.py')
    with open(finit, 'r', encoding="utf-8") as fpnt:
        for line in fpnt:
            if line.startswith('__version__'):
                return literal_eval(line.split('=', 1)[1].strip())
    return '0.0.1-alpha'

setup(
    name='gen2mkt',
    version=get_version(),
    description='Connector between parcel generation and parcel market models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Horizon-LEAD/Conn_Gen2Mkt',
    keywords='lead, parcel, generation, market',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    install_requires=[
        'networkx',
        'pandas',
        'pyshp',
        'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'gen2mkt=gen2mkt.__main__:main'
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/Horizon-LEAD/Conn_Gen2Mkt/issues',
        'Source': 'https://github.com/Horizon-LEAD/Conn_Gen2Mkt',
    }
)
