from setuptools import find_namespace_packages
from setuptools import setup

setup(
    name='kits23',
    packages=find_namespace_packages(include=["kits23*"]),
    version='0.1.0',
    description='',
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'opencv-python',
        'nibabel',
        'requests',
        'argparse',
        'tqdm',
        "pytest",
        'Surface-Distance-Based-Measures @ git+https://github.com/deepmind/surface-distance.git',
        'SimpleITK'
    ],
    entry_points={
        'console_scripts': [
            'kits23_download_data = kits23.entrypoints:download_data_entrypoint',  # todo!
            'kits23_compute_metrics = kits23.evaluation.entry_point:main'
        ]
    },
)
