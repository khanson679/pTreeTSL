import argparse
import csv
import pyparsing as pp
import pandas as pd
import random
import re
import scipy
import time
import warnings

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from numpy.random import rand
from scipy.special import xlogy
from scipy.optimize import minimize
from scipy import stats

well_formed_cache = {}

class Tree():
    def __init__(self, label, features, children=None):
        # Name of node
        self.label = label
        # Set of strings
        self.features = features
        # List of trees
        if not children:
            self.children = []
        else:
            self.children = children

        self.str_cache = None

    def __eq__(self, other):
        return (
            self.label == other.label and
            self.features == other.features and
            self.children == other.children
        )

    def __str__(self, depth=1):
        """
        This returns a printable string, not the string used for TSL
        """
        child_string = ''
        tabs = ''.join(['  '] * depth)

        if self.children:
            child_string = '\n{} {}'.format(
                tabs,
                '\n{}'.format(tabs).join(
                child.__str__(depth + 1) for child in self.children
            )
        )
        result = "{} {}\n{}".format(
            self.label, 
            self.features,
            child_string
        )

        return result

    def __repr__(self):
        return self.__str__()

    def get_child_string(self):
        """
        This returns the string representation of the node's children.
        Not currently used, but could be helpful if you want to implement
        the SL-2 functions as string functions over child strings.
        """
        return ' '.join('{}{}'.format(child.label, child.features)
            for child in self.children
        )

    def project(self, target_features):
        """
        Kicks off recursive process for projecting a tree
        """
        return self.project_helper(target_features)[0]

    def project_helper(self, target_features):
        """
        Recursively projects a tree
        """
        if target_features.intersection(self.features):
            return [
                Tree(
                    label = self.label,
                    features = self.features,
                    children = [
                        projected_child
                        for child in self.children
                        for projected_child in child.project_helper(target_features)
                    ]
                )]
        else:
            return [
                projected_child
                for child in self.children
                for projected_child in child.project_helper(target_features)
            ]

    def check_well_formed(self, sl_function):
        """
        Checks whether a tree is well formed given an SL function.
        """
        # Optimization 1: Exit early if we find any illicit child
        if not str(self) in well_formed_cache:
            wf = sl_function(self)
            
            for child in self.children:
                wf = wf and child.check_well_formed(sl_function)
                if not wf:
                    break

            well_formed_cache[str(self)] = wf
        else:
            wf = well_formed_cache[str(self)]
        return wf

    def count_child_features(self, feature):
        """
        Helper function that counts the number of children that
        contain some feature.
        """
        return sum(feature in child.features for child in self.children)

    def get_all_features(self):
        '''
        :return: returns Set of all features contained in the Tree
        '''
        if not self.children:
            return self.features

        return self.features.union(*[child.get_all_features() for child in self.children])

    @classmethod
    def from_str(cls, string, feature_dict):
        """
        Reads in a string representation of a tree and constructs a tree
        object. proj_dict is used to map node names to features.
        """
        lpar = pp.Suppress('(')
        rpar = pp.Suppress(')')
        name = pp.Word(pp.alphanums + "_" + "-")
        expr = pp.Forward()

        expr <<= (
            pp.Group(lpar + name + rpar) | 
            pp.Group(lpar + name + expr + pp.ZeroOrMore(expr) + rpar)
        )

        parse = expr.parseString(string)

        return cls.build_tree_from_parse(parse[0], feature_dict)

    @classmethod
    def build_tree_from_parse(cls, parse, feature_dict):
        """
        Recursively builds a tree from a pyparsing parse
        """
        parent, *children = parse
        # handling for uncoded features
        if parent in feature_dict:
            features = feature_dict[parent]
        else:
            features = parent

        if not children:
            tree = Tree(
                label = parent,
                features = features
            )

        else:
            child_trees = [
                cls.build_tree_from_parse(child, feature_dict) for child in children
            ]
            tree = Tree(
                label = parent,
                features = features,
                children = child_trees
            )

        return tree

class TSL2_Grammar:
    def __init__(self, functions: list, proj_dict: dict):
        # needs final edits for how features will be loaded
        '''
        :param functions: list of functions to check projected trees with
        :param proj_dict: dictionary of probabilities
        may need to be edited with changes to feature representation and projection
        '''
        self.functions = functions
        self.proj_dict = proj_dict

        self.proj_p_helper_cache = {}

    def clear_caches(self):
        self.proj_p_helper_cache = {}

    def is_grammatical(self, tree: Tree):
        # Optimization 2: Fail early
        val = True
        for f in self.functions:
            val = (val and tree.check_well_formed(f))
            if not val:
                break
        return val

    def score_dataset(self, dataset):
        probs = []
        for label, tree, score in dataset:
            model_score = self.p_grammatical(tree)
            probs.append((label, tree, score, model_score, abs(model_score - score)))

        return probs

    def projection_p(self, tree: Tree):
        '''
        Returns a list of projections of tree and their probabilities
        :param tree: Tree
        :return: List(Tuple(Tree, float)), list of projections and their probabilities
        '''
        projection_probs = self.projection_p_helper(tree)
        return [(Tree(label="TOP_FISH", features="TOP_FISH", children=proj), prob) for
                proj, prob in projection_probs]

    def projection_p_helper(self, tree: Tree):
        '''
        :param tree: Tree
        :return: List(Tuple(Tree, float)), list of projections and their probabilities
        '''
        if str(tree) in self.proj_p_helper_cache:
            return self.proj_p_helper_cache[str(tree)]

        prob = self.proj_dict[tree.features]

        # base case
        if not tree.children:
            # Optimization 3: Prune impossible projections
            result = []
            if prob != 1:
                result.append(([], 1-prob))
            if prob != 0:
                result.append(([tree], prob))

        else:
            # Get all projections of each child
            sub_projections = [self.projection_p_helper(child) for child in tree.children]
            # Get every permutation of child projections
            possible_children = TSL2_Grammar.child_product(sub_projections)

            result = []

            for children, val in possible_children:
                # projections including node and all possible projected children
                if prob != 0:
                    result.append(
                        ([Tree(tree.label, tree.features, children=children)], prob * val)
                    )
                # projections without node
                if prob != 1:
                    result.append((children, (1 - prob) * val))

        self.proj_p_helper_cache[str(tree)] = result

        return result

    def get_sse(self, corpus_probs, normalize=True):
        sse = 0
        for i, (_, tree, p) in enumerate(corpus_probs):
            #print(i)
            # start = time.time()
            #print(tree)
            #print(grammar.p_grammatical(tree))
            sse += (self.p_grammatical(tree) - p)**2
            # end = time.time()
            # print("Took {}".format(end - start))
        if normalize:
            sse /= len(corpus_probs)
        return sse

    @staticmethod
    def evaluate_proj(proj_probs, grammar, params, fixed_params, corpus_probs, beta):
        # may be rewritten to use a more motivated method than sse
        '''
        Straightforward adaptation from Connor's pTSL code
        :param proj_probs: Tuple(float) probabilities in tuple equal length(params)
        :param grammar
        :param params: List(Str) Keys for proj_dict
        :param corpus_probs: Tuple(Tree, float) trees and their acceptability/grammaticality probability
        :param beta: float, regularization penalty
        :return:
        '''
        for i, param in enumerate(params):
            grammar.proj_dict[param] = proj_probs[i]

        for i, param in enumerate(fixed_params):
            grammar.proj_dict[param] = 1

        grammar.clear_caches()

        sse = grammar.get_sse(corpus_probs)
        bias =  beta * sum(xlogy(proj_probs, proj_probs) + xlogy(1 - proj_probs, 1 - proj_probs))
        print(proj_probs)
        print(sse)
        score = sse - bias
        return score

    @staticmethod
    def train(sl_functions, corpus_file, feature_file, feature_key, 
              free_params_subs, fixed_params, betas, outfile, name, itr, is_categorical):
        '''
        Straightforward adaptation from Connor's pTSL code
        :param sl_functions: List(Function), a list of functions to be used to check grammaticality
        :param corpus_file: Str, location of corpus_file
        :param feature_file: Str, location of feature_file
        :param free_params: List(Str), dictionary keys from proj_dict whose probabilities will be optimized
        :param fixed_params: List(Str), dictionary keys from proj_dict to set to 1
        :param beta: float, regularization constant
        :return:
        '''

        print("Reading feature file")
        features = read_feature_file(feature_file, feature_key)
        print("Reading training data")
        corpus_scores = read_corpus_file(corpus_file, features, is_categorical)

        # Fix this later
        if not free_params_subs:
            free_params = list(set(features.values()))
        else:
            with open(free_params_subs) as f:
                reader = csv.reader(f)
                regex = '|'.join(re.escape(x[0]) for x in reader)
            free_params = [x for x in list(set(features.values())) if re.search(regex, x)]


        if not fixed_params:
            fixed_params = []
        else:
            with open(fixed_params) as f:
                reader = csv.reader(f)
                regex = '|'.join(re.escape(x[0]) for x in reader)
            fixed_params = [x for x in list(set(features.values())) if re.search(regex, x)]

        free_params = list(set(free_params) - set(fixed_params))
        print("Free params: {}".format(free_params))
        print("Fixed params: {}".format(fixed_params))

        # create bounds
        # instead of limiting bound for fixed value, I removed it completely from the parameter
        bounds = [(0, 1) for _ in range(len(free_params))]

        if outfile:
            header = ['name', 'beta', 'itr', 'sse', 'obj', 'item', 'model_score', 'human_score']
            header.extend(free_params)

            with open(outfile, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        for beta in betas:
            for i in range(itr):
                # randomly initialize parameter - this will be the input
                proj_probs = rand(len(free_params))

                grammar = TSL2_Grammar(sl_functions, defaultdict(int))

                print("Beginning training")
                # run the minimize function
                proj_res = minimize(TSL2_Grammar.evaluate_proj,
                                    proj_probs,
                                    bounds=bounds,
                                    method='L-BFGS-B',
                                    args=(grammar, free_params, fixed_params, corpus_scores, beta))
                print(proj_res)

                if outfile:
                    fitted_probs = proj_res.x
                    obj = proj_res.fun
                    sse = grammar.get_sse(corpus_scores)
                    row_base = [name, beta, i, sse, obj]
                    rows = []
                    for (item, tree, human_score) in corpus_scores:
                        model_score = grammar.p_grammatical(tree)
                        rows.append(row_base + [item, model_score, human_score] + list(fitted_probs))

                    with open(outfile, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)

        for i, param in enumerate(free_params):
            print("{}: {}".format(param, proj_res.x[i]))

        return grammar

    def p_grammatical(self, tree: Tree):
        '''
        :param tree: Tree to check
        :param proj_dict: Features to project
        :return: Probability tree is grammatical under this instantiation of grammar
        '''
        projection_probs = self.projection_p(tree)
        p_g = sum([prob for proj, prob in projection_probs if prob > 0 and self.is_grammatical(proj)])
        return p_g

    @staticmethod
    def child_product(child_projections):
        '''
        Returns all possible combinations of child projections
        :param child_projections: list of tuples List(Tuple(Tree, float)) of children projections and probs
        :return: List(Tuple(List(Tree), float)) new list of (children, probability) tuples of possible children the
        parent node has to worry about and their probability
        '''
        first, *rest = child_projections
        # base case, return list
        if not rest:
            return first
        projections_products = []
        for r_projection in TSL2_Grammar.child_product(rest):
            # for all the other projections in the recursive call
            for f_projection in first:
                # for all the projections currently being evaluated 'first'
                # make a new projection/probability pair which is the two lists appended and their probs multiplied
                projections_products.append((f_projection[0]+r_projection[0], f_projection[1]*r_projection[1]))
        # remove 0 prob projections to prevent wasteful memory usage
        return [(projection, prob) for projection, prob in projections_products if prob != 0]

def read_corpus_file(corpus_file, features, is_categorical):
    '''
    :param corpus_file: file of sentence trees and probabilities
    :param features: dictionary of labels: features
    :is_categorical: boolean flag for categorical runs
    :return: list of tuples contain (tree, judgment score) pairs
    '''
    with open(corpus_file, encoding="utf-8-sig") as c_file:
        corpus_df = pd.read_csv(c_file, encoding="utf-8-sig")

    feature_binary = {
        "WH.whe.non.sh": 1,
        "WH.whe.non.lg": 1,
        "WH.whe.isl.sh": 1,
        "WH.whe.isl.lg": 0,
        "RC.sub.non.sh": 1,
        "RC.sub.non.lg": 1,
        "RC.sub.isl.sh": 1,
        "RC.sub.isl.lg": 0,
        "RC.wh.non.sh": 1,
        "RC.wh.non.lg": 1,
        "RC.wh.isl.sh": 1,
        "RC.wh.isl.lg": 0,
        "WH.sub.non.sh": 1,
        "WH.sub.non.lg": 1,
        "WH.sub.isl.sh": 1,
        "WH.sub.isl.lg": 0,
        "WH.np.non.sh": 1,
        "WH.np.non.lg": 1,
        "WH.np.isl.sh": 1,
        "WH.np.isl.lg": 0,
        "RC.adj.non.sh": 1,
        "RC.adj.non.lg": 1,
        "RC.adj.isl.sh": 1,
        "RC.adj.isl.lg": 0,
        "WH.adj.non.sh": 1,
        "WH.adj.non.lg": 1,
        "WH.adj.isl.lg": 0,
        "WH.adj.isl.sh": 1,
        "RC.np.non.sh": 1,
        "RC.np.non.lg": 1,
        "RC.np.isl.sh": 1,
        "RC.np.isl.lg": 0
    }

    if is_categorical:
        return [(row['item'], Tree.from_str(row['tree'], features), feature_binary[row['condition']])
                for _, row in corpus_df.iterrows()]

    return [(row['item'], Tree.from_str(row['tree'], features), row['score']) for _, row in corpus_df.iterrows()]

def read_feature_file(feature_file, feature_key, entry_key="symbol"):
    '''
    :param feature_file: location of feature_file
    :param feature_key: the column name in the file that contains features
    :param entry_key: the column name in the file that contains symbols
    :return: A dictionary mapping symbols to features
    '''
    with open(feature_file, encoding="utf-8-sig") as f_file:
        feature_df = pd.read_csv(f_file, encoding="utf-8-sig")

    # features = {row[entry_key]: set(row[feature_key].split(' ')) for _, row in feature_df.iterrows()}
    features = {row[entry_key]: row[feature_key] for _, row in feature_df.iterrows()}
    return features

def write_results_file():
    pass

def check_wh(tree):
    """
    SL-2 function for checking for wh feature violations
    """
    if '+wh' in tree.features:
        return tree.count_child_features('-wh') == 1
    else:
        return tree.count_child_features('-wh') == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "training_file", type=str, 
        help="The path to the .csv containing the training data"
    )
    parser.add_argument(
        "feature_file", type=str,
        help="The path to the .csv containing the mapping from lexical symbols"
             "to features."
    )
    parser.add_argument(
        "--feature_key", type=str, default="features",
        help="The column name in the feautres file that contains features."
    )
    parser.add_argument(
        "--free_params", type=str, default=None,
        help="File containing list of featural substrings to choose included features. If this is "
             "not specified, probabilities will be learned for all symbols."
    )
    parser.add_argument(
        "--fixed_params", type=str, default=None,
        help="The parameters that will always project (probability of projection fixed to 1)."
    )
    parser.add_argument(
        "--beta", nargs="+", type=float, default=(0,),
        help="Regularization penalty. Higher values penalize fitted values "
             "that deviate from the prior more strongly."
    )
    parser.add_argument(
        '--outfile', type=str, default=None,
        help="Path to save model results to."
    )
    parser.add_argument(
        '--name', type=str, default='model',
        help='Model name'
    )
    parser.add_argument(
        '--itr', type=int, default=10,
        help='Number of times to re-run optimization'
    )
    parser.add_argument(
        '--categorical', action="store_true",
        help='Number of times to re-run optimization'
    )
    args = parser.parse_args()
    trained_grammar = TSL2_Grammar.train([check_wh],
        args.training_file, args.feature_file, args.feature_key, args.free_params, 
        args.fixed_params, args.beta, args.outfile, args.name, args.itr, args.categorical
    )

    # features = read_feature_file(args.feature_file, args.feature_key)
    # corpus_scores = read_corpus_file(args.training_file, features, False)
    # proj_dict = defaultdict(int)
    # proj_dict_contents = {
    #     'whether:: +T -C': 1,
    #     '-D -wh': 1,
    #     'that:: +T +wh -C': 1,
    #     'e:: +T +wh -C': 1,
    #     'if:: +T -C': 1,
    #     '+C -N': 1,

    #     'e:: +T -C': 0.25,
    #     'that:: +T -C': 0.12
    # }
    # for key, value in proj_dict_contents.items():
    #     proj_dict[key] = value

    # grammar = TSL2_Grammar([check_wh], proj_dict)
    # scores = grammar.score_dataset(corpus_scores)

    # list_to_save = [(item, human_score, model_score, delta) for item, _, human_score, model_score, delta in scores]
    # df = pd.DataFrame(list_to_save)
    # df.to_csv(args.outfile, header=False, index=False)

    # python src/tree.py data/training_data.csv data/ptreetsl_lexicon.csv --feature_key features --free_params C wh --beta 1
