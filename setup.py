from setuptools import setup, find_packages

setup(
    name="clustering-app",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "streamlit>=1.25.0",
        "pytest>=7.4.0",
        "joblib>=1.3.1",
    ],
    python_requires=">=3.9",
)
