import unittest
import random
from mep.genetics.gene import VariableGene, OperatorGene
from mep.genetics.chromosome import Chromosome
import numpy as np


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
