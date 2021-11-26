import setuptools


setuptools.setup(
    name="crisp-sir-sibyl",
    version="0.0.2",
    author="Fabio Mazza",
    author_email="fab4mazz@gmail.com",
    description="Implementation of CRISP for the SIR model",
    url="https://sibyl-team.github.io/",    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ],
    license="MPL 2.0",
    packages=["crisp_sir"],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "numba",
    ]
)
