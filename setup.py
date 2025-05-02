# setup.py
from setuptools import setup, find_packages

setup(
    name="deep-reinforcement-learning",
    version="1.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "gymnasium>=0.29.1"
    ],
    author="Mattia Beltrami",
    author_email="mattia.beltrami@studio.unibo.it",
    description="A modular, high-level Python package for Deep Reinforcement Learning, designed to simplify the implementation and study of DRL algorithms for students, researchers, and enthusiasts.",
    long_description=open("README.md").read() if open("README.md", errors="ignore") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/beltromatti/deep-reinforcement-learning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)