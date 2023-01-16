from subprocess import Popen

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tmsneurosim",
    version="0.0.1",
    author="Torge Worbs",
    author_email="torgeworbs@gmail.com",
    description="...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TorgeW/TMS-Neuro-Sim",
    project_urls={
        "Bug Tracker": "https://gitlab.gwdg.de/worbs/tms-simulation/issues",
    },
    install_requires=[
        'neuron>=8.0.2',
        'numpy>=1.16',
        'scipy>=1.2',
        'simnibs>=4.0',
        'tqdm>=4.64.0',
        'matplotlib>=3.5.1',
        'h5py>=3.6.0',
        'vtk>=9.0.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=['cells_hoc']),
    package_data={
        "tmsneurosim": ["nrn/coil_recordings/*", "nrn/mechanisms/*", "nrn/mechanisms/*/*", "nrn/mechanisms/*/.libs/*",
                        "nrn/cells/cells_hoc/*/LICENSE", "nrn/cells/cells_hoc/*/morphology/*/*.asc"],
    },
    python_requires=">=3.8",
)

n = Popen(['nrnivmodl'], cwd='tmsneurosim/nrn/mechanisms')
n.wait()
