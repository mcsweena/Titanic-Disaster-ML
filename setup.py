try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Titanic Machine Learning',
    'author': 'Andy McSweeney',
    'url': '',
    'download_url': '',
    'author_email': 'andy.mcsweeney91@gmail.com',
    'version': '0.1',
    'install_requires': ['TBC'],
    'packages': ['TBC'],
    'scripts': [],
    'name': 'titanic problem'
}

setup(**config)
