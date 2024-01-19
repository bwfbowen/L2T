import setuptools 

with open('README.md', 'r') as f:
  long_description = f.read()
  
setuptools.setup(
  name='l2t',
  version='0.1',
  authors = [{ 'name': "Bowen Fang", 'email': "bf2504@columbia.edu" },],
  description="",
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=setuptools.find_packages(),
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  python_requires='>=3.6',
  py_modules=['src'],
)
  
