"""A setuptools based setup module.
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

    finit = join(here, 'src', 'tour2copert', '__init__.py')
    with open(finit, 'r', encoding="utf-8") as fpnt:
        for line in fpnt:
            if line.startswith('__version__'):
                return literal_eval(line.split('=', 1)[1].strip())
    return '0.0.0'

setup(
    name='tour2copert',
    version=get_version(),
    description='A sample Python project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Horizon-LEAD/Conn_Sched2COPERT',
    keywords='lead, parcel, tour, formation, copert',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    install_requires=[
        'python-dotenv',
        'numpy',
        'pandas',
        'xlsxwriter'
    ],
    entry_points={
        'console_scripts': [
            'tour-2-copert=tour2copert.__main__:main'
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/Horizon-LEAD/Conn_Sched2COPERT',
        'Source': 'https://github.com/Horizon-LEAD/Conn_Sched2COPERT',
    }
)
