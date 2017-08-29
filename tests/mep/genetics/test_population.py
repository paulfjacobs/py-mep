import unittest
import random
import numpy as np
from mep.genetics.population import Population
from mep.genetics.chromosome import Chromosome

class TestPopulation(unittest.TestCase):
    """
    Test the Population class.
    """

    def test_random_tournament_selection(self):
        """
        Test the random_tournament_selection(...)
        """
        # make it so this repeatable
        random.seed(0)

        # construct the population
        num_examples = 5
        num_features = 7
        population = Population(np.zeros((num_examples, num_features)), [], 1, 1, 1, 1, 1, 1, 1)

        # confirm the number of feature variables (not critical for this test)
        self.assertEqual(num_features, population.num_feature_variables)

        # test the tournament selection; not that it randomly chooses the not as good chromosome
        min_chromosome, max_chromosome = Chromosome([], []), Chromosome([], [])
        min_chromosome.error = 1
        max_chromosome.error = 2
        population.chromosomes = [min_chromosome, max_chromosome]
        self.assertEqual(max_chromosome, population.random_tournament_selection(1))

    def test_larger_random_tournament_selection(self):
        """
        Test the random_tournament_selection(...)
        """
        # make it so this repeatable
        random.seed(0)

        # construct the population
        num_examples = 5
        num_features = 7
        population = Population(np.zeros((num_examples, num_features)), [], 1, 1, 1, 1, 1, 1, 1)

        # test the tournament selection; not that it randomly chooses the not as good chromosome
        min_chromosome, max_chromosome = Chromosome([], []), Chromosome([], [])
        min_chromosome.error = 1
        max_chromosome.error = 2
        population.chromosomes = [min_chromosome, max_chromosome]
        self.assertEqual(min_chromosome, population.random_tournament_selection(10))