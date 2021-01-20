# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
	name='diygrad',
	version='1.0.0',
	description='',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/chr1sbradley/diygrad',
	author='Christopher Bradley',
	author_email='declarationcb@gmail.com',
	keywords='machine, learning, machinelearning,ai,micrograd,torch',
	packages=['diygrad'],
	python_requires='>=3.8.5',
	install_requires=['numpy', 'requests'],
	extras_require={
		'gpu': ["pyopencl", "six"],
		'testing': ["torch","tqdm"]
	},
	# install_requires
	# If there are data files included in your packages that need to be
	# installed, specify them here.
	# package_data={ 'sample': ['model.pickle'], }
)