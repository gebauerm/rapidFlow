# rapidFlow

This is a project, that tries to accelerate micro research projects by providing a richer functionality for the
already known hpyerparameter optimization library [optuna](https://github.com/optuna/optuna). The code of optuna is not
modified, it is incorporated into rapidFlow to provide richer evaluation and easy parallel processing.

# Getting Started

## Prerequisites

* Python >= 3.7
* [PyTorch](https://pytorch.org/)

## Install

rapidFlow is build upon Pytorch, so make sure you have PyTorch installed.

1. From Pip
Install package with: \
    `pip install rapidflow`

2. With cloned repository
Install package with:
\
    `pip install -e /src`

# Collaboration

|Branche|Purpose|
|-------|-------|
|main|production state|
|feature| a new feautre|
|hotfix|hotfix as there are no bugfixes as everything is created from master|

The desired workflow is [github flow](https://githubflow.github.io/).
Meaning that:
    * we can deploy from master at any time
    * nothing gets deployed without a PR and its review
    * we have no releases or release branches

This way we maintain:
    * fast responses to features or bugs and continouus delivery
    * easy workflow
    * fast developer feedback

more to come!

## Pipelines

PR:
* deplyos into develop and performs integration tests
    * 2 PRs are created at the same time --> their tests and deployment is scheduled over Runners in github or jenkins
    * 1 PR is merged --> if there is a dependency a merge conflict arises, which gets reslved by a new commit in the 2nd PR --> triggers pipeline


# TODO:

* move experiment library to another repo
* experiments in docker container with gpu? (or singularity)
* test on multiple gpus
* testing and propper doku
* significance testing

# Acknowledgments
Feel free to contribute. If you use this repository please cite with:

        @misc{rapidFlow_geb,
        author = {Gebauer, Michael},
        title = {rapidFlow},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/gebauerm/model_storage}},
        }


# Author

[elysias](https://github.com/gebauerm)
