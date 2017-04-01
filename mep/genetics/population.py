from mep.genetics.chromosome import Chromosome
import random


class Population(object):
    """
    A collection of chromosomes.
    """

    def __init__(self, data_matrix, targets, num_constants, constants_min, constants_max, constants_prob,
                 feature_variable_prob, num_genes, num_chromosomes, operators_prob):
        """
        Build a randomly constructed chromosome.

        :param data_matrix: the data matrix; rows are feature vectors; comes from the data set; it is (n, m) where "n"
        is the number of examples and "m" is the number of features.
        :type data_matrix: np.matrix
        :param targets: the targets; equal to the number of examples (n)
        :type targets: list
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
        :param num_genes: how many genes
        :type num_genes: int
        :param num_chromosomes: how many chromosomes to use
        :type num_chromosomes: int
        :param operators_prob: the probability that a given gene is an operator
        :type operators_prob: float
        """
        # set the variables
        self.data_matrix = data_matrix
        self.targets = targets
        self.num_constants = num_constants
        self.constants_min = constants_min
        self.constants_max = constants_max
        self.constants_prob = constants_prob
        self.feature_variable_prob = feature_variable_prob
        self.num_feature_variables = self.data_matrix.shape[1]
        self.num_genes = num_genes
        self.num_chromosomes = num_chromosomes
        self.operators_prob = operators_prob

        # the chromosomes
        self.chromosomes = None

    def initialize(self):
        """
        Initialize the random chromosomes.
        """
        # generate the random chromosomes
        self.chromosomes = [Chromosome.generate_random_chromosome(self.num_constants, self.constants_min,
                                                                  self.constants_max, self.constants_prob,
                                                                  self.feature_variable_prob,
                                                                  self.num_feature_variables, self.num_genes,
                                                                  self.operators_prob)
                            for _ in range(self.num_chromosomes)]

        # evaluate
        # TODO: this could be done in parallel
        for chromosome in self.chromosomes:
            chromosome.evaluate(self.data_matrix, self.targets)

        # sort the chromosomes
        self.chromosomes.sort()

    def random_tournament_selection(self, tournament_size):
        """
        Randomly select (tournament_size) chromosomes and return the best one.
        :param tournament_size: the size of the tournament
        :type tournament_size: int
        :return: the
        """
        # TODO: Check for bad tournament size
        best_chromosome = None
        for _ in range(tournament_size):
            chromosome = random.choice(self.chromosomes)
            if best_chromosome is None or chromosome.error < best_chromosome.error:
                best_chromosome = chromosome

        return best_chromosome

    def next_generation(self):
        """
        Advance to the next generation.
        """
        # TODO: populate

