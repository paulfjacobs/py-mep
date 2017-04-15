import logging
import numpy as np
from collections import deque
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
        self.logger = logging.getLogger(self.__class__.__name__)

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

    def mutate(self, gene_mutation_prob, num_constants, constants_min, constants_max, constants_prob,
               feature_variable_prob, num_feature_variables, num_genes, operators_prob):
        """
        Mutate the chromosome. Works by going through and randomly mutating genes and then constants.
        :param gene_mutation_prob: probability to mutate a given gene
        :type gene_mutation_prob: float
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
        :return: nothing
        """
        # the probabilities are all the same for generating a random chromosome; therefore let's construct
        # a random chromosome and then (effectively) do a uniform crossover where a "mutate" means that we
        # take the new chromosome's gene/constants
        # TODO: Should we have these variables set in the chromosome then?
        # TODO: maybe just pass in this random chromosome then?
        random_chromosome = Chromosome.generate_random_chromosome(num_constants, constants_min,
                                                                  constants_max, constants_prob,
                                                                  feature_variable_prob,
                                                                  num_feature_variables, num_genes,
                                                                  operators_prob)

        # go through mutating genes;
        for gene_index in range(len(self.genes)):
            # decide if we are going to mutate this gene
            if random() <= gene_mutation_prob:
                # mutated; therefore grab the corresponding gene from the random chromosome
                self.genes[gene_index] = random_chromosome.genes[gene_index]

        # go through mutating constants;
        for constants_index in range(len(self.constants)):
            # decide if we are going to mutate this gene
            if random() <= gene_mutation_prob:
                # mutated; therefore grab the corresponding constant from the random chromosome
                self.constants[constants_index] = random_chromosome.constants[constants_index]

    def __str__(self):
        return "Chromosome({}, {})".format(self.genes, self.constants)

    def pretty_string(self, stop_at_best=True):
        """
        Output in a program like format. First show the constants. Then one line per gene.
        :return: the program
        :rtype: str
        """
        # first we show the constants
        program = "CONSTANTS = [{}]\n".format(",".join([str(c) for c in self.constants]))

        # now show each gene on a separate line
        for gene_index, gene in enumerate(self.genes):
            gene_str = gene.__str__()
            if type(gene) == VariableGene:
                gene_str = gene.pretty_string()
            elif type(gene) == OperatorGene:
                # TODO: Push this logic into the gene; the only tricky part is the operator lambda; we will probably
                # need to replace the lambda with a larger object
                if gene.operation == Chromosome.operator_lambdas[0]:
                    op = "+"
                elif gene.operation == Chromosome.operator_lambdas[1]:
                    op = "-"
                elif gene.operation == Chromosome.operator_lambdas[2]:
                    op = "*"
                gene_str = "PROGRAM[{}] {} PROGRAM[{}]".format(gene.address1, op, gene.address2)
            program += "{}:{}\n".format(gene_index, gene_str)

            if self.best_gene_index == gene_index and stop_at_best:
                return program

        # if we want to print the full program
        return program

    def prune(self):
        """
        Trim out the unused genes. NOTE: This "breaks" the chromosomes as it is going to change how many genes are
        in the program. Only do this once we have finished evolving the program.
        """
        # the best gene index is going to be the last line of the program; since the genes never reference genes
        # beyond it then we just proceed back to the top and remove any which haven't been referenced; we determine
        # this via a BFS type search

        # the genes that are in use -- i.e. that will be kept;
        gene_indices_in_use = set()
        visited = set()

        # start from best gene index
        genes_indices_to_visit = deque()
        genes_indices_to_visit.appendleft(self.best_gene_index)
        gene_indices_in_use.add(self.best_gene_index)

        while len(genes_indices_to_visit) > 0:
            # the index to visit
            gene_index = genes_indices_to_visit.pop()

            # mark as visited
            visited.add(gene_index)

            # check the addresses on the gene if it is an operator
            gene = self.genes[gene_index]
            if type(gene) == OperatorGene:
                genes_indices_to_visit.appendleft(gene.address1)
                genes_indices_to_visit.appendleft(gene.address2)
                gene_indices_in_use.add(gene.address1)
                gene_indices_in_use.add(gene.address2)
                self.logger.debug("At gene index {} which references {} and {}".format(gene_index,
                                                                                       gene.address1, gene.address2))

        # now remove any genes that aren't used
        gene_indices_in_use = list(gene_indices_in_use)
        gene_indices_in_use.sort()
        self.logger.debug("All gene indices in use {}".format(gene_indices_in_use))
        self.genes = [self.genes[i] for i in gene_indices_in_use]

        # TODO: This could be done in the list comprehension but it is clearer to just do another pass
        # re-map the address to the new index
        for gene in self.genes:
            if type(gene) == OperatorGene:
                gene.address1 = gene_indices_in_use.index(gene.address1)
                gene.address2 = gene_indices_in_use.index(gene.address2)

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
