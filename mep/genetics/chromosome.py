import logging
import numpy as np
from mep.genetics.gene import Gene, VariableGene, OperatorGene
from random import random, randint, choice


class Chromosome(object):
    """
    Level above Gene. Each chromosome is a fixed number of genes and constants. We can think of a chromosome as a
    program where each gene is a line of code in the program. Genes can reference the result of other genes by their
    line number (address) in the program/chromosome. The overall fitness (i.e. error) is equal to the best of the genes.
    """

    # valid operators
    operator_lambdas = [lambda a, b: a + b,  # +
                        lambda a, b: a - b,  # -
                        lambda a, b: a * b]  # *

    def __init__(self, genes, constants):
        """
        Initialize.
        :param genes: the genes in the chromosome.
        :type genes: list of Gene
        :param constants: the constants
        :type constants: list of float
        """
        # self.logger = logging.getLogger(self.__class__)

        # core genes and constants lists
        self.genes = genes
        self.constants = constants

        # track the best found error and the associated gene
        self.error = float('inf')
        self.best_gene_index = -1

    @classmethod
    def generate_random_chromosome(cls, num_constants, constants_min, constants_max, constants_prob,
                                   feature_variable_prob, num_feature_variables, num_genes, operators_prob):
        """
        Build a randomly constructed chromosome.

        :param num_constants: how many constants to have
        :type num_constants: int
        :param constants_min: the min range of the constants
        :type constants_min: float
        :param constants_max: the max range of the constants
        :type constants_max: float
        :param constants_prob: the probability that a given gene is a constant
        :type constants_prob: float
        :param feature_variable_prob: the probability that a given gene is a feature variable
        :type feature_variable_prob: float
        :param num_feature_variables: how many features we have
        :type num_feature_variables: int
        :param num_genes: how many genes
        :type num_genes: int
        :param operators_prob: the probability that a given gene is an operator
        :type operators_prob: float
        """
        # generate num_constants random constants between (constants_min, constants_max)
        constants = [random() * (constants_max - constants_min) + constants_min for _ in range(num_constants)]

        # going to generate the genes
        genes = []

        # the very first gene must be either a constant or a feature variable; that is because the operators can only
        # reference the addresses of previous genes so if there have been no previous ones then the operators have
        # nothing to operate on.
        sum_prob = constants_prob + feature_variable_prob
        if random() * sum_prob <= feature_variable_prob:
            # randomly decide on a feature variable index
            genes.append(VariableGene(randint(0, num_feature_variables - 1), is_feature=True))
        else:
            # randomly decide on a constants
            genes.append(VariableGene(randint(0, num_constants - 1), is_feature=False))

        # now we generate the other genes based upon the probabilities passed in
        for gene_index in range(1, num_genes):
            prob = random()
            if prob <= operators_prob:
                # randomly choose valid addresses; randomly choose an operator
                genes.append(OperatorGene(choice(Chromosome.operator_lambdas),
                                          randint(0, gene_index - 1), randint(0, gene_index - 1)))
            elif prob <= operators_prob + feature_variable_prob:
                genes.append(VariableGene(randint(0, num_feature_variables - 1), is_feature=True))
            else:
                genes.append(VariableGene(randint(0, num_constants - 1), is_feature=False))

        # construct and return the chromosome
        return Chromosome(genes, constants)

    def evaluate(self, data_matrix, targets):
        """
        Evaluate the various genes.

        :param data_matrix: the data matrix; rows are feature vectors; comes from the data set; it is (n, m) where "n"
        is the number of examples and "m" is the number of features.
        :type data_matrix: np.matrix
        :param targets: the targets; equal to the number of examples (n)
        :type targets: list
        """
        num_examples = data_matrix.shape[0]
        eval_matrix = np.zeros((len(self.genes), num_examples))
        for gene_index, gene in enumerate(self.genes):
            # compute the error for this gene; if it is the best we have found then update
            error = gene.evaluate(gene_index, eval_matrix, data_matrix, self.constants, targets)
            if error < self.error:
                self.error = error
                self.best_gene_index = gene_index

    def __str__(self):
        return "Chromosome({}, {})".format(self.genes, self.constants)

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        """
        Less-than used by sort(...)

        :param other:
        :type other: Chromosome
        :return:
        """
        return self.error < other.error