from setuptools import setup,find_packages
setup(
    name="VEHICLE_INSURANCE",
    version="0.1.0"
    author="ANSHUMAN PARIDA"
    author_email="anshu989856@gmail.com"
    description="ML PIPELINE FOR VEHICLE INSURANCE"
    long_description=open("README.md",encoding="utf-8").read()
    long_description_content_type="text/markdown",
    url="https://github.com/Anshu989856/PROJECT"
    packages=find_packages(exclude=["tests*","notebooks*","docs*"]),
    include_package_data=True
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ], 
    classifiers=[
        Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
     python_requires=">=3.8",
)