from setuptools import setup

setup(name='turnit',
      version='0.1',
      description='Orient STL files to minimize supports',
      long_description=open('README.md').read(),
      url='http://github.com/nickc92/turnit',
      author='Nick C',
      author_email='nickcholy@gmail.com',
      license='MIT',
      packages=['turnit'],
      install_requires=['matplotlib', 'scipy', 'numpy', 'stl', 'rtree'],
      scripts=['bin/turnit'],
      zip_safe=False)
