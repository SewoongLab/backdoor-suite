# robust-ml-suite
## tl;dr
A module-based repository for testing and evaluating backdoor attacks and defenses. For information on experiments and testing [click here](##installation).

---
## Introduction
As third party and federated machine learning models become more popular, so, too, will attacks on their training processes. In particular, this repository focuses on a new class of 'backdoor' attacks in which an attacker 'poisons' or tampers with training data so that at evaluation time, they have control over the class that the model outputs.

With this repository we hope to provide a ubiquitous testing and evaluation platform to standardize the settings under which these attacks and their subsequent defenses are considered, pitting relevant attack literature against developed defenses. In this light, we welcome any contributions or suggested changes to the repository. 

In the rest of this document we detail (1) [how the repo works](##in-the-repo) (2) [how to run an experiment](##installation), and (3) [how to contribute](##adding-content). Please don't hesitate to file a GitHub issue or reach out [Rishi Jha](rishijha.com) for any issues or requests!

---

## In this repo
This repo is split into three main folders: `experiments`, `modules` and `schemas`. The `experiments` folder (as described in more detail [here](#installation)) contains subfolders and `.toml` configuration files on which an experiment may be run. The `modules` folder stores source code for each of the subsequent part of an experiment. These modules take in specific inputs and outputs as defined by their subseqeunt `.toml` documentation in the `schemas` folder. 

In particular, each module defines some specific task in the attack-defense chain. As mentioned earlier, each module has explicitly defined inputs and outputs that, we hope, facilitate the addition of attacks and defenses with diverse requirements (i.e., training loops or representations). As discussed [here]() we hope that researchers can add their own modules or expand on the existing `base` modules.

### Existing modules:
1. `base_trainer`: Configured to poison and train ResNet on the CIFAR-10 dataset.
1. `base_rep_saver`: Configured to extract representations from ResNet on poisoned CIFAR-10 data.
1. `base_defense`: Configured to implement a defense based on the class representations on poisoned CIFAR-10. At the moment implements three defenses: PCA, k-means, and SPECTRE.
1. `base_utils`: Utility module, used by the base modules.

More documentation can be found in the `schemas` folder.

---
## Installation
### Prerequisites:
The prerequisite packages are stored in `requirements.txt` and can be installed using pip:
```
pip install -r requirements.txt
```
Or conda:
```
conda install --file requirements.txt
```
Note that the requirements encapsulate our testing enviornments and may be unnecessarily tight! Any relevant updates to the requirements are welcomed.

### Submodules:
This library relies heavily on git submoduling to natively support other repositories, so, after cloning it is required to pull all git submodules, which can be done like so:
```
git submodule update --init --recursive
``` 

## Running An Experiment
### Setting up:
To initialize an experiment, create a subfolder in the `experiments` folder with the name of your experiment:
```
mkdir experiments/[experiment name]
```
In that folder initialize a config file called `[experiment name].toml`. An example can be seen here: `experments/example/example.toml`.

The `.toml` file should contain references to the modules that you would like to run with each relevant field as defined by its documentation in `schemas/[module name]`. This file will serve as the configuration file for the entire experiment. As a convention the output for module **n** is the input for module **n + 1**.

```
[module_name_1]
output=...
field2=...
...
fieldn=...

[module_name_2]
input=...
output=...
...
fieldn=...

...

[module_name_k]
input=...
field2=...
...
fieldn=...
```

### Running a module:
At the moment, all modules must be manually run using:
```
python modules/[module name]/run_experiment.py [experiment name] [module name]
```
The module will automatically pick up on the configuration provided by the file. 

As an example, to run the initial training regime for the example experiment one could run:
```
python modules/base_trainer/run_experiment.py example base_trainer
```
More module documentation can be found in the `schemas` folder.

---

## Adding Content:
One of the goals of this project is to develop a ubiquitous testing and validation framework for backdoor attacks. As such, we appreciate and welcome all contributions ranging fron structural changes to additional attacks and defenses.

The fastest way to add an attack, defense, or general feature to this repository is to submit a pull request, however, time permitting, the repository maintainer is available to help [contact](rishijha.com).

### Publishing your own module
Finishing...

### Adding to existing modules
Finishing...

---
## Planned Features
### General:
* Automated test runner based on `.toml` configs 
### Attacks:
* [Label Consistent Backdoor Attacks](https://arxiv.org/abs/1912.02771)
* [Hidden Trigger Backdoor Attacks](https://arxiv.org/abs/1910.00033)
### Defenses:
* [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531)