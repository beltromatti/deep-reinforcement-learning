# setup.py
from setuptools import setup, find_packages

setup(
    name="dqn-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "gymnasium>=0.29.1"
    ],
    author="Mattia Beltrami",
    author_email="your.email@unibo.it",
    description="A modular Deep Q-Network (DQN) package for reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/dqn-rl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)