# pTreeTSL

## About

This repository contains code used in 

> Torres, C., Hanson, K., Graf, T., & Mayer, C. (2023). [Modeling island effects with probabilistic tier-based strictly local grammars over trees.](https://sites.socsci.uci.edu/~cjmayer/papers/torres_et_al_pTreeTSL_SCiL_2023.pdf) _Proceedings of the Society for Computation in Linguistics._ Vol. 4. Article 15.

This code fits a pTSL grammar to a data set that consists of minimalist dependency trees with associated Likert scores.

## Structure of the repository

### src/

The code use in the paper.

* `tree.py`: This is the code responsible for fitting the model. Its use is described in detail below.
* `produce_training_file.R`: A script to combine Likert ratings from Sprouse et al. (2016) with the corresponding dependency trees.
* `analyze_results.R`: A script for visualizing the results of the model, computing correlations, etc.

### data/

This folder contains the following subfolders:
* `fixed_params`: Configuration files specifying which parameters should have projection probabilities fixed to 1.
* `free_params`: Configuration files specifying which parameters should be fit to data.
* `lexicon`: Files specifying mappings between lexical labels and syntactic features.
* `sprouse_data`: Likert ratings from Sprouse et al. (2016)
* `training_data`: Files containing Likert rating - dependency tree pairs.
* `trees`: Annotated trees.

The files with `_agg` correspond to averaged Likert ratings rather than individual participants' ratings. The files with `_filtered` include only the island violations used in the paper, while the files without `_filtered` have the full set of island types from Sprouse et al. (2016). The `no_wh` files correspond to simulations where nodes with `wh` features were treated as free parameters.

The paper used the aggregated, filtered data with `wh` features fixed to 1.

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

The command used to run the model in the paper was:

`python src/tree.py data/training_data/training_data_agg_filtered.csv data/lexicon/ptreetsl_lexicon_phonetic.csv --feature_key features --free_params data/free_params/free_params_fixed.csv --fixed_params data/fixed_params/fixed_params_wh.csv --outfile results/results.csv`
                       
