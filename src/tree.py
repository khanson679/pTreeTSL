import argparse
import csv
import pyparsing as pp
import pandas as pd
import random
import scipy
import warnings

from collections import defaultdict
from dataclasses import dataclass
from numpy.random import rand
from scipy.optimize import minimize

@dataclass
class Word:
    form: str
    features: set

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

    def __str__(self, depth=1):
        """
        This returns a printable string, not the string used for TSL
        """
        child_string = ''
        tabs = ''.join(['  '] * depth)

        if self.children:
            child_string = '\n{}{}'.format(
                tabs,
                '\n{}'.format(tabs).join(
                child.__str__(depth + 1) for child in self.children
            )
        )
        return "{}{}\n{}".format(
            self.label, 
            self.features,
            child_string
        )

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
        return sl_function(self) and all(child.check_well_formed(sl_function) for child in self.children)

    def count_child_features(self, feature):
        """
        Helper function that counts the number of children that
        contain some feature.
        """
        return sum(feature in child.features for child in self.children)

    def get_probs(self, feature_dict: dict):
        '''
        Currently set to work for the case where each feature set contains a single item
        :param feature_dict: dictionary of feature (str): probabilities (float)
        :return: probability (float)
        likely to be tweaked
        '''

        #if len(set(proj_dict[self.label]).intersection(self.features))>1:
        #    warnings.warn("Multiple matching features.")

        for feature, prob in feature_dict.items():
            if feature in self.features:
                return prob
        return 0

    def get_all_features(self):
        '''
        :return: returns Set of all features contained in the Tree
        '''
        if not self.children:
            return self.features
        return self.features.union(*[child.get_all_features for child in self.children])



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
            features = {parent}

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

    def p_project(self, feature_probs, label_probs):
        # assumes all probabilities 0 < p < 1
        # and assumes only one feature in the set for now
        '''
        :param feature_probs: dictionary of features and probability to project
        :param label_probs: dictionary of labels and probability to project
        :return: Tree()
        '''

        if self.label in label_probs:
            x = random.random()
            if x < label_probs[self.label]:
                return [
                    Tree(
                        label=self.label,
                        features=self.features,
                        children=[
                            projected_child
                            for child in self.children
                            for projected_child in child.p_project(feature_probs, label_probs)
                        ]
                    )]

        for feature in self.features:
            if feature in feature_probs:
                x = random.random()
                if x < feature_probs[feature]:
                    return [
                        Tree(
                            label = self.label,
                            features = self.features,
                            children = [
                                projected_child
                                for child in self.children
                                for projected_child in child.p_project(feature_probs, label_probs)
                            ]
                        )]
        else:
            return [
                projected_child
                for child in self.children
                for projected_child in child.p_project(feature_probs, label_probs)
            ]


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

    def is_grammatical(self, tree: Tree):
        return all([tree.check_well_formed(f) for f in self.functions])

    def projection_p(self, tree: Tree):
        '''
        :param tree: Tree
        :return: List(Tuple(Tree, float)), list of projections and their probabilities
        '''
        return [(Tree(label="TOP_FISH", features={"TOP_FISH"}, children=proj), prob) for
                proj, prob in self.projection_p_helper(tree)]

    def projection_p_helper(self, tree: Tree):
        '''
        :param tree: Tree
        :return: List(Tuple(Tree, float)), list of projections and their probabilities
        '''
        # base case
        if not tree.children:
            prob = tree.get_probs(self.proj_dict)
            return [(proj, prob) for proj, prob in [([tree], prob), ([], 1-prob)] if prob != 0]

        # probability of being projected, function to be tweaked
        prob = tree.get_probs(self.proj_dict)
        sub_projections = [self.projection_p_helper(child) for child in tree.children]
        possible_children = TSL2_Grammar.child_product(sub_projections)

        new_children = []
        for children, val in possible_children:
            # projections including node and all possible projected children
            if prob != 0:
                new_children.append(([Tree(tree.label, tree.features, children=children)], prob * val))
            # projections without node
            if prob != 1:
                new_children.append((children, (1 - prob) * val))
        return new_children

    @staticmethod
    def evaluate_proj(proj_probs, grammar, params, corpus_probs, prior, beta):
        # may be rewritten to use a more motivated method than sse
        '''
        Straightforward adaptation from Connor's pTSL code
        :param proj_probs: Tuple(float) probabilities in tuple equal length(params)
        :param grammar
        :param params: List(Str) Keys for proj_dict
        :param corpus_probs: Tuple(Tree, float) trees and their acceptability/grammaticality probability
        :param prior: scipy.stats.beta, a prior on the bernoulli RVs
        :param beta: float, regularization penalty
        :return:
        '''
        for i, param in enumerate(params):
            grammar.proj_dict[param] = proj_probs[i]

        sse = 0
        for i, (tree, p) in enumerate(corpus_probs):
            print(i)
            sse += (grammar.p_grammatical(tree) - p)**2 - beta*prior.pdf(p)

        return sse

    @staticmethod
    def train(sl_functions, corpus_file, feature_file, feature_key, free_params, prior_params, beta):
        '''
        Straightforward adaptation from Connor's pTSL code
        :param sl_functions: List(Function), a list of functions to be used to check grammaticality
        :param corpus_file: Str, location of corpus_file
        :param feature_file: Str, location of feature_file
        :param free_params: List(Str), dictionary keys from proj_dict whose probabilities will be optimized
        :param prior: scipy.stats.beta, a prior on the bernoulli RVs
        :param beta: float, regularization constant
        :return:
        '''
        prior = scipy.stats.beta(*prior_params)

        print("Reading feature file")
        features = read_feature_file(feature_file, feature_key)
        print("Reading training data")
        corpus_scores = read_corpus_file(corpus_file, features)

        if not free_params:
            free_params = list(set([x for feat_set in features.values() for x in feat_set]))

        # create bounds
        # instead of limiting bound for fixed value, I removed it completely from the parameter
        bounds = [(0, 1) for _ in range(len(free_params))]

        # randomly initialize parameter - this will be the input
        proj_probs = rand(len(free_params))

        grammar = TSL2_Grammar(sl_functions, {})

        def callback(X):
            print(X)

        print("Beginning training")
        # run the minimize function
        proj_res = minimize(TSL2_Grammar.evaluate_proj,
                            proj_probs,
                            bounds=bounds,
                            method='L-BFGS-B',
                            args=(grammar, free_params, corpus_scores, prior, beta),
                            callback=callback)

        return grammar

    def p_grammatical(self, tree: Tree):
        '''
        :param tree: Tree to check
        :param proj_dict: Features to project
        :return: Probability tree is grammatical under this instantiation of grammar
        '''
        return sum([prob for proj, prob in self.projection_p(tree) if self.is_grammatical(proj)])

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
            # for all the other projections in the recursive call 'projection_powerset(rest)'
            for f_projection in first:
                # for all the projections currently being evaluated 'first'
                # make a new projection/probability pair which is the two lists appended and their probs multiplied
                projections_products.append((f_projection[0]+r_projection[0], f_projection[1]*r_projection[1]))
        # remove 0 prob projections to prevent wasteful memory usage
        return [(projection, prob) for projection, prob in projections_products if prob != 0]


def read_corpus_file(corpus_file, features):
    '''
    :param corpus_file: file of sentence trees and probabilities
    :param features: dictionary of labels: features
    :return: list of tuples contain (tree, judgment score) pairs
    '''
    with open(corpus_file, encoding="utf-8-sig") as c_file:
        corpus_df = pd.read_csv(c_file, encoding="utf-8-sig")

    return [(Tree.from_str(row['tree'], features), row['zscores']) for _, row in corpus_df.iterrows()]

def read_feature_file(feature_file, feature_key, entry_key="symbol"):
    '''
    :param feature_file: location of feature_file
    :param feature_key: the column name in the file that contains features
    :param entry_key: the column name in the file that contains symbols
    :return: A dictionary mapping symbols to features
    '''
    with open(feature_file, encoding="utf-8-sig") as f_file:
        feature_df = pd.read_csv(f_file, encoding="utf-8-sig")

    features = {row[entry_key]: set(row[feature_key].split(' ')) for _, row in feature_df.iterrows()}
    return features

def check_wh(tree):
    """
    SL-2 function for checking for wh feature violations
    """
    if 'wh+' in tree.features:
        return tree.count_child_features('wh-') == 1
    else:
        return tree.count_child_features('wh-') == 0

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
        help="The projection probabilities that will be learned. If this is "
             "not specified, probabilities will be learned for all symbols."
    )
    parser.add_argument(
        "--prior_params", nargs="+", type=float, default=[1, 1],
        help="Parameters (a, b) for prior beta distribution over projection "
             "probabilities. This is specified as a pair of numbers separated "
             "by a space."
    )
    parser.add_argument(
        "--beta", type=float, default=0,
        help="Regularization penalty. Higher values penalize fitted values "
             "that deviate from the prior more strongly."
    )

    args = parser.parse_args()
    trained_grammar = TSL2_Grammar.train([check_wh],
        args.training_file, args.feature_file, args.feature_key, args.free_params, 
        args.prior_params, args.beta
    )
