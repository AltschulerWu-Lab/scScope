from setuptools import setup

install_requires = [
    'numpy>=1.14.0',
    'pandas>=0.21.0',
    'scipy',
    'scikit-learn>=0.18.0',
    'phenograph',
    'tensorflow>=1.2.1'
]
setup(name='scScope_cpu',
      version='0.1.5',
      description='scScope is a deep-learning based approach for single cell RNA-seq analysis. ',
      url='https://github.com/AltschulerWu-Lab/scScope',
      author=['Yue Deng', 'Feng Bao'],
      author_email='yue.deng@ucsf.edu',
      license='Apache License 2.0',
      packages=['scscope'],
      install_requires=install_requires,
      zip_safe=False)
