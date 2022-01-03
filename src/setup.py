from setuptools import setup


with open("../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='rapidflow',
    version='0.1',
    description='rapidFlow - A framework to perform micro experimentation fast with easy scaling.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    author='Michael Gebauer',
    author_email='gebauerm23@gmail.com',
    url='https://github.com/gebauerm/rapidFlow',
    download_url='https://github.com/gebauerm/rapidFlow/archive/refs/tags/v0.1-alpha.tar.gz',
    packages=['rapidflow'],
    install_requires=[
        "optuna==2.9.1",
        "click==8.0.1", "scikit-learn==0.24.2", "scipy==1.7.0", "networkx==2.5.1", "psycopg2-binary",
        "docker==5.0.3", "pytest==6.2.5", "pandas==1.3.5", "torch", "tqdm==4.62.3"],
    dependency_links=[""],
    python_requires=">=3.7",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ]
)
