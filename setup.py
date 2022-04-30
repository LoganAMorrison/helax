from setuptools import setup
from setuptools import find_packages


setup(
    name="helax",
    version="0.1.0",
    description="Helicity Amplitudes with Jax",
    author="Logan A. Morrison",
    author_email="loganmorrison99@gmail.com",
    packages=find_packages(exclude=["notebooks"]),
    package_data={},
    install_requires=["jax>=0.3.4", "jaxlib>=0.3.2", "numpy>=1.22.3", "flax>=0.4.1"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
