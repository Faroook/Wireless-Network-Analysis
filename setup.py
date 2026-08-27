from setuptools import setup, find_packages

setup(
    name="wireless-network-analysis",
    version="0.1.0",
    author="Sharif Umar Farook",
    description="Wireless Network Performance Analysis Tool",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
    ],
    python_requires=">=3.8",
)