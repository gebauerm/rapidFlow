from setuptools import find_packages, setup


URL = "https://github.com/gebauerm/rapidFlow"
__version__ = "0.1.4"

install_requires = [
        "optuna==2.9.1",
        "click==8.0.1", "scikit-learn==0.24.2", "scipy==1.7.0", "networkx==2.5.1", "psycopg2-binary",
        "docker==5.0.3", "pandas==1.3.5", "torch", "tqdm==4.62.3"],

test_require = ["pytest==6.2.5"]


setup(
    name='rapidflow',
    version=__version__,
    description='rapidFlow - A framework to perform micro experimentation fast with easy scaling.',
    license='MIT',
    author='Michael Gebauer',
    author_email='gebauerm23@gmail.com',
    url='https://github.com/gebauerm/rapidFlow',
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"test": test_require},
    dependency_links=[""],
    python_requires=">=3.7",
    include_package_data=True,
)
