import logging
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
        self.logger = logging.getLogger(self.__class__)

        self.genes = genes
        self.constants = constants

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
