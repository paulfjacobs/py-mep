import unittest
from mep.genetics.gene import VariableGene
import numpy as np


class TestVariableGene(unittest.TestCase):
    """
    Tests for the variable gene.
    """

    def test_basic_constant(self):
        """
        Simple check of a constant gene with just 1 gene in the chromosome.
        """
        # construct
        constant_index = 0
        gene = VariableGene(constant_index, is_feature=False)

        # simple eval matrix; 1 gene in a chromosome, 3 examples, 2 constants
        num_examples = 2
        num_genes = 1
        num_features = 3

        # create
        constants = [1., 2.]
        eval_matrix = np.zeros((num_genes, num_examples))
        data_matrix = np.zeros((num_examples, num_features))

        # expected; only one gene and it is going to be using the first constant;
        gene_index = 0
        expected_eval_matrix = np.matrix([[constants[constant_index], constants[constant_index]]])

        # run the evaluate
        gene.evaluate(gene_index, eval_matrix, data_matrix, constants)
        self.assertTrue(np.array_equal(expected_eval_matrix, eval_matrix))

