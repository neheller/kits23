from setuptools import setup


setup(
    name='kits21',
    version='0.0.1',
    description='',
    zip_safe=False,
    install_requires=[
        'numpy',
        "scipy",
        "scikit-image",
        "opencv-python",
        'nibabel',
        'requests',
        'argparse',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'kits23_download_data = kits23.entrypoints:download_data_entrypoint',
            'kits23_compute_metrics = kits23.entrypoints:compute_metrics_entrypoint'
        ]
    }
)
