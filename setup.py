import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eecr",
    version="0.0.1",
    author="Vito Janko",
    author_email="vito.janko@ijs.si",
    description="Tools for the optimization of energy in a context recognition task",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/VitoJanko/Energy-Efficient-Context-Recognition",
    packages=['eecr'],
    #packages=setuptools.find_packages(),
    #package_dir={"": "eecr"},
    #py_modules=["eeoptimizer","eeutility","altmethods"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        #"Intended Audience :: Science/Research,"
        #"Intended Audience :: Developers",
        #"Natural Language :: English",
        #"Topic :: Scientific/Engineering :: Artificial Intelligence"
        #"Topic :: Software Development",

    ],
    python_requires='>=3',
    install_requires=["scikit-learn", "deap", "pandas", "numpy", "matplotlib"],
)