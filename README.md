# AL NLP
Active Learning framework for Natural Language Processing of pathology reports.

# Repository content

This repository implements an active learning loop for natural language processing of pathology reports related to NCI Pilot 3 project. It implements two methods for embedding extraction of the unstructured text: 1) bag-of-words with dimensional reduction methods, and 2) pre-trained BERT model. Deterministic and Bayesian classifiers are available to predict attributes contained in the pathology reports.

# Repository structure

```
al_nlp
│   README.md  
|   acquisition_functions.py   # contains acquisitions functions available to use
|   datasets.py   # class to store dataset information
|   design.py   # implements active learning loop
│
└───classifiers/    # code for all classification methods
│   │   bayesian/   # code for Linear Bayesian methods (Bayesian models naturally provides uncertainty quantification)
│   │   bootstrapping/  # code for bootstrapping-based methods (uncertainty is obtained from multiple models)
|   |   deterministic/   # codes for deterministic methods (use softmax probability as uncertainty)
│   
└───feature_extraction/   # codes for BERT and Bag-of-Words feature extraction
│   
└───experiments/  # scripts to run active learning experiments
|
└───outputs/   # store results and reports (pdf files) of the experiments
|
└───data/   # store downloaded datasets
|
└───path_reports_preprocessing/  # scripts to pre-process pathology reports (data not present in this repository)
|
└───utils/  # auxiliary codes

```

# Software requirements
- Operating System: Windows or MacOS
- Python 3.7 or higher
- Python packages required:
  - transformers (https://github.com/huggingface/transformers)
  - pymc3 3.8 (https://github.com/pymc-devs/pymc3)
  - pytorch 1.3+
  - nltk  3.4+
  - theano 1.0
  - scikit-learn 0.20+
  - matplotlib 3.0+
  - pytorch 1.3+
  - pandas 
  - ...

For easy installation, we made a conda enviroment available in `environment.yml`. To recreate the same environment used to develop the code, simply do:

> conda env create -f environment.yml

# How to run it
The folder named `experiments` contains a set of python scripts demonstrating how to execute the active learning loop for a given set of classification methods and acquisition functions for a particular dataset. For instance, `experiment_001.py` run the active learning loop for 4 logistic regression models, each one using a different acquisition function. The dataset used is the well-known [20-NewsGroup](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) dataset. In the `ActiveLearningLoop`'s execute method, the user can inform the percentages of data initially used for training, the size of test set, and the how many new samples will be selected to be labeled at every iteration of the active learning loop. After the execution, a report with all the results and plots will be stored in the `outputs` folder. A sub-folder with the same name as the python script will be created (`experiment_001` in this case). A pdf with plots will be placed in this sub-folder.

> python experiment_001.py

# Authors

- André R. Gonçalves (goncalves1@llnl.gov)
- Hiranmayi Ranganathan (ranganathan2@llnl.gov)
- Braden C. Soper (soper3@llnl.gov)
- Pryiadip Ray (ray34@llnl.gov)
- Ana Paula Sales (deoliveirasa1@llnl.gov)

# Release number

LLNL-CODE-797271
