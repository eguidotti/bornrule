import os
from setuptools import setup


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            return line.split('"' if '"' in line else "'")[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="bornrule",
    packages=["bornrule", "bornrule.sql", "bornrule.torch"],
    version=get_version("bornrule/__init__.py"),
    description="Supervised classification with Born's rule.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    python_requires='>=3',
    install_requires=['scikit-learn', 'pandas', 'scipy', 'numpy'],
    author='Emanuele Guidotti',
    author_email='emanuele.guidotti@unine.ch',
    license='GPLv3',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    url='https://github.com/eguidotti/bornrule',
    project_urls={
        'Documentation': 'https://github.com/eguidotti/bornrule',
        'Source': 'https://github.com/eguidotti/bornrule',
        'Tracker': 'https://github.com/eguidotti/bornrule/issues',
    },
)
