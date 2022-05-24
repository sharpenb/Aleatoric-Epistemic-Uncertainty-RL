from setuptools import setup, find_packages

setup(name='uncertainty-rl',
      version='0.1',
      description='Uncertainty RL',
      author='anonymous',
      author_email='anonymous@hotmail.com',
      packages=['src'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'tqdm',
                        'sacred', 'deprecation', 'pymongo', 'pytorch-lightning>=0.9.0rc2', 'seml'],
      zip_safe=False)
