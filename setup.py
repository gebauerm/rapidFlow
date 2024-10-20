from setuptools import find_packages, setup
import rapidflow as package


setup(
    name='rapidflow',
    version=package.__version__,
    description=package.long_description,
    license='MIT',
    author='Michael Gebauer',
    author_email='gebauerm23@gmail.com',
    url='https://github.com/gebauerm/rapidFlow',
    download_url=f'{package.URL}/archive/{package.__version__}.tar.gz',
    packages=find_packages(),
    install_requires=package.install_requires,
    extras_require={"test": package.test_require},
    dependency_links=[""],
    python_requires=">=3.7",
    include_package_data=True,
)
