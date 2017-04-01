import unittest
import random
from mep.genetics.gene import VariableGene, OperatorGene, Gene
from mep.genetics.chromosome import Chromosome
import numpy as np


class MockedGene(Gene):
    def __init__(self, error_to_return):
        """
        Initialize.
        :param error_to_return: what to return in the evaluate
        :type error_to_return: float
        """
        self.error_to_return = error_to_return

    def evaluate(self, gene_index, eval_matrix, data_matrix, constants, targets):
        """
        Simple mocked version.
        """
        return self.error_to_return


class TestChromosome(unittest.TestCase):
    """
    Tests for the chromosome.
    """

    def test_basic_random_construction(self):
        """
        Basic example of a construction.
        """
        # set the seed to keep it reproducible
        random.seed(0)

        # create the chromosome
        num_genes = 2
        num_constants = 1
        chromosome = Chromosome.generate_random_chromosome(num_constants=num_constants, constants_min=1,
                                                           constants_max=10, constants_prob=0.2,
                                                           feature_variable_prob=0.3,
                                                           num_feature_variables=2, num_genes=num_genes,
                                                           operators_prob=0.5)

        # confirm the number of genes and constants match what we expect
        self.assertEquals(num_genes, len(chromosome.genes))
        self.assertEquals(num_constants, len(chromosome.constants))

        # the first gene has to be a variable gene; in particular it is this one
        self.assertEquals(VariableGene(0, is_feature=False), chromosome.genes[0])

        # the 2nd gene can be a variable or an operator; in this case it is the below
        self.assertEquals(OperatorGene(Chromosome.operator_lambdas[1], 0, 0), chromosome.genes[1])

        # verify constant
        self.assertAlmostEquals(8.599796663725433, chromosome.constants[0])

    def test_evaluate(self):
        """
        Basic test of the evaluate method.
        """
        # construct mocked genes
        genes = [MockedGene(10), MockedGene(1)]

        # construct chromosome
        chromosome = Chromosome(genes, constants=[1, 2, 3])

        # evaluate
        chromosome.evaluate(np.zeros((2, 2)), targets=[20, 30])

        # confirm the genes
        self.assertEqual(genes[1], genes[chromosome.best_gene_index])
        self.assertEqual(genes[1].error_to_return, chromosome.error)

    def test_sort(self):
        """
        Test the sort mechanism.
        """
        # construct the chromosomes and test sorting them (by error)
        min_chromosome, mid_chromosome, max_chromosome = Chromosome([], []), Chromosome([], []), Chromosome([], [])
        min_chromosome.error = 1
        mid_chromosome.error = 2
        max_chromosome.error = 3
        chromosomes = [mid_chromosome, max_chromosome, min_chromosome]
        expected_chromosomes = [min_chromosome, mid_chromosome, max_chromosome]

        # do the sort and verify
        chromosomes.sort()
        self.assertEqual(expected_chromosomes, chromosomes)
