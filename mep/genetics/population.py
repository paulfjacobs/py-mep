from mep.genetics.chromosome import Chromosome
import random
import copy


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

        # TODO: take in
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1

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

    def one_cut_point_crossover(self, parent1, parent2):
        """
        Construct two offspring chromosomes from the parents. We determine the crossover point so that we
        take the first genes up to that point from parent1/parent2 and then we switch.
        :param parent1: one parent chromosome
        :type parent1: Chromosome
        :param parent2: the other parent chromosome
        :type parent2: Chromosome
        :return: two offsprings
        :rtype: (Chromosome, Chromosome)
        """
        # construct the genes and constants for the offsprings from the parents
        offspring1 = Chromosome([], [])
        offspring2 = Chromosome([], [])

        # determine the crossover point;
        cutting_point = random.randint(0, self.num_genes)

        # TODO: copy the genes
        # copy over the genes; first half and now the 2nd half (from the other chromosome)
        offspring1.genes = parent1.genes[:cutting_point] + parent2.genes[cutting_point:]
        offspring2.genes = parent2.genes[:cutting_point] + parent1.genes[cutting_point:]

        # same thing with the constants
        cutting_point = random.randint(0, self.num_constants)

        # copy over the constants; first half and now the 2nd half
        offspring1.constants = parent1.constants[:cutting_point] + parent2.constants[cutting_point:]
        offspring2.constants = parent2.constants[:cutting_point] + parent1.constants[cutting_point:]

        return offspring1, offspring2

    def next_generation(self):
        """
        Advance to the next generation.
        """
        for _ in range(0, len(self.chromosomes), 2):
            # select parents
            chromosome1 = self.random_tournament_selection(2)
            chromosome2 = self.random_tournament_selection(2)

            # crossover
            if random.random() < self.crossover_prob:
                offspring1, offspring2 = self.one_cut_point_crossover(chromosome1, chromosome2)
            else:
                # offspring are copies of the parents
                offspring1 = copy.copy(chromosome1)
                offspring2 = copy.copy(chromosome2)

            # mutate (potentially) offspring
            offspring1.mutate(self.mutation_prob, self.num_constants, self.constants_min,
                              self.constants_max, self.constants_prob,
                              self.feature_variable_prob,
                              self.num_feature_variables, self.num_genes,
                              self.operators_prob)
            # TODO: evaluate
            offspring2.mutate(self.mutation_prob, self.num_constants, self.constants_min,
                              self.constants_max, self.constants_prob,
                              self.feature_variable_prob,
                              self.num_feature_variables, self.num_genes,
                              self.operators_prob)

            # TODO: fill in
