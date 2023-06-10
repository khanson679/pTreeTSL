# pTreeTSL

## About

This repository contains code used in 

> `Torres, C., Hanson, K., Graf, T., & Mayer, C. (2023). [Modeling island effects with probabilistic tier-based strictly local grammars over trees.](https://sites.socsci.uci.edu/~cjmayer/papers/torres_et_al_pTreeTSL_SCiL_2023.pdf) _Proceedings of the Society for Computation in Linguistics._ Vol. 4. Article 15.`

This code fits a pTSL grammar to a data set that consists of minimalist dependency trees with associated Likert scores.

## Structure of the repository

### src/

The code use in the paper.

* `tree.py`: This is the code responsible for fitting the model. Its use is described in detail below.
* `produce_training_file.R`: A script to combine Likert ratings from Sprouse et al. (2016) with the corresponding dependency trees.
* `analyze_results.R`: A script for visualizing the results of the model, computing correlations, etc.

### data/

This folder contains training and test data sets, as well as configuration files that determine which parameters the model will fit and which will have fixed values.

Details about the dependency tree annotation scheme are given in [data/annotation.md](data/annotation.md).

### figs/

Some figures used in the paper.

### results/

Results of the model.

## Running `tree.py`

`tree.py` can be run from the command line. It expects the following arguments.
* `training_file` (required): The path to the .csv containing the training data. See the training data sets for examples of the required format.
* `feature_file` (required): The path to the .csv containing the mapping from lexical symbols to features. See the training data sets for examples of the required format.
* `feature_key` (optional): The column name in the feautres file that contains features.
* `free_params` (optional): File containing list of featural substrings to choose included features. If this is not specified, probabilities will be learned for all symbols.
* `fixed_params` (optional): The parameters that will always project (probability of projection fixed to 1).
* `beta` (optional): Regularization penalty. Higher values will force learned parameters to be closer to 0 or 1 (i.e. more categorical). Defaults to 0 (no regularization). Not used in the paper.
* `outfile` (optional): Path to save model results to.
* `name` (optional): Model name. Used in output files.
* `itr` (optional): Number of times to re-run optimization).

Some examples of running `tree.py` from the command line can be found in [commands.txt].
                       
