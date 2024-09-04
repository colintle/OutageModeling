from setuptools import setup

setup(
    name='Outage Model API',
    version='0.1.0',
    py_modules=['outage_model'],
    install_requires=[
        'Click==8.1.7',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.3.1',
        'matplotlib==3.9.0',
        'torch-geometric==2.5.3',
        'torch-optim==0.0.4',
    ],
    entry_points={
        'console_scripts': [
            'outage-model = outage_model.main:cli',
        ],
    },
)

