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

    def test_basic_feature_gene(self):
        """
        Simple check of a feature/input gene with just 1 gene in the chromosome.
        """
        # construct
        feature_index = 1
        gene = VariableGene(feature_index, is_feature=True)

        # simple eval matrix; 1 gene in a chromosome, 3 examples, 2 constants
        num_examples = 2
        num_genes = 1
        num_features = 3

        # create
        constants = [1., 2.]
        eval_matrix = np.zeros((num_genes, num_examples))
        data_matrix = np.zeros((num_examples, num_features))

        # set the data matrix for the feature that we care about
        data_matrix[0, feature_index] = 5.
        data_matrix[1, feature_index] = 7.

        # expected; only one gene and it is going to be using the first constant;
        gene_index = 0
        expected_eval_matrix = np.matrix([[data_matrix[0, feature_index], data_matrix[1, feature_index]]])

        # run the evaluate
        gene.evaluate(gene_index, eval_matrix, data_matrix, constants)
        self.assertTrue(np.array_equal(expected_eval_matrix, eval_matrix))

    def test_constant_and_feature_gene(self):
        """
        Intermix constant and feature genes.
        """
        # construct
        feature_index = 1
        constant_index = 0
        constant_gene = VariableGene(constant_index, is_feature=False)
        feature_gene = VariableGene(feature_index, is_feature=True)

        # simple eval matrix; 1 gene in a chromosome, 3 examples, 2 constants
        num_examples = 2
        num_genes = 2
        num_features = 3

        # create
        constants = [1., 2.]
        eval_matrix = np.zeros((num_genes, num_examples))
        data_matrix = np.zeros((num_examples, num_features))

        # set the data matrix for the feature that we care about
        data_matrix[0, feature_index] = 5.
        data_matrix[1, feature_index] = 7.

        # expected;
        expected_eval_matrix = np.matrix([[data_matrix[0, feature_index], data_matrix[1, feature_index]],
                                          [constants[constant_index], constants[constant_index]]])

        # run the evaluate
        feature_gene.evaluate(0, eval_matrix, data_matrix, constants)
        constant_gene.evaluate(1, eval_matrix, data_matrix, constants)
        self.assertTrue(np.array_equal(expected_eval_matrix, eval_matrix))

