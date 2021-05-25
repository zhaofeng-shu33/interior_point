from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()
    
if __name__ == '__main__':
    setup(name = 'sdp_ip',
          version = '0.1',
          description = 'semi-definite programming for the recovery of stochastic block model (interior point)',
          author = 'zhaofeng-shu33',
          author_email = '616545598@qq.com',
          url = 'https://github.com/zhaofeng-shu33/interior_point',
          maintainer = 'zhaofeng-shu33',
          maintainer_email = '616545598@qq.com',
          long_description = long_description,
          long_description_content_type="text/markdown",          
          install_requires = ['scipy'],
          license = 'Apache License Version 2.0',
          py_modules = ['sdp_ip'],
          classifiers = (
              "Development Status :: 4 - Beta",
              "Programming Language :: Python :: 3.8",
              "Operating System :: OS Independent",
          ),
          )