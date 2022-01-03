from setuptools import setup

setup(
    name='rapidflow',
    version='0.1',
    description='A place to perform micro experimentation fast with easy scaling.',
    license='MIT',
    author='Michael Gebauer',
    author_email='gebauerm23@gmail.com',
    url='https://github.com/gebauerm/rapidFlow',
    packages=['rapidflow'],
    install_requires=[
        "optuna==2.9.1",
        "click==8.0.1", "scikit-learn==0.24.2", "scipy==1.7.0", "networkx==2.5.1", "psycopg2-binary",
        "docker==5.0.3", "pytest==6.2.5", "pandas==1.3.5"],
    dependency_links=[""],
    python_requires=">=3.7",
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    ]
)
