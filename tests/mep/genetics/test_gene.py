import unittest
from mep.genetics.gene import VariableGene, OperatorGene
import numpy as np


class TestGene(unittest.TestCase):
    """
    Tests for the genes.
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
        targets = [0] * num_examples

        # expected; only one gene and it is going to be using the first constant;
        gene_index = 0
        expected_eval_matrix = np.array([[constants[constant_index], constants[constant_index]]])

        # run the evaluate
        error = gene.evaluate(gene_index, eval_matrix, data_matrix, constants, targets)
        self.assertTrue(np.array_equal(expected_eval_matrix, eval_matrix))
        self.assertEqual((1. - 0) + (1. - 0), error)

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
        targets = [0] * num_examples

        # set the data matrix for the feature that we care about
        data_matrix[0, feature_index] = 5.
        data_matrix[1, feature_index] = 7.

        # expected; only one gene and it is going to be using the first constant;
        gene_index = 0
        expected_eval_matrix = np.array([[data_matrix[0, feature_index], data_matrix[1, feature_index]]])

        # run the evaluate
        error = gene.evaluate(gene_index, eval_matrix, data_matrix, constants, targets)
        self.assertTrue(np.array_equal(expected_eval_matrix, eval_matrix))
        self.assertEqual((5. - 0.) + (7. - 0.), error)

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
        targets = [0] * num_examples

        # set the data matrix for the feature that we care about
        data_matrix[0, feature_index] = 5.
        data_matrix[1, feature_index] = 7.

        # expected;
        expected_eval_matrix = np.array([[data_matrix[0, feature_index], data_matrix[1, feature_index]],
                                          [constants[constant_index], constants[constant_index]]])

        # run the evaluate
        feature_error = feature_gene.evaluate(0, eval_matrix, data_matrix, constants, targets)
        constant_error = constant_gene.evaluate(1, eval_matrix, data_matrix, constants, targets)
        self.assertTrue(np.array_equal(expected_eval_matrix, eval_matrix))

    def test_operator_gene_basic(self):
        """
        This is a test of the operator gene. We need at least two genes as the operator needs to be able to reference
        another gene evaluation.
        """
        # construct; using the same address on both sides of the operator; in other words we will be adding the previous
        # gene (at 0) to itself
        address_index = 0
        gene = OperatorGene(lambda a, b: a + b, address1=address_index, address2=address_index)

        # simple eval matrix; 2 gene in a chromosome, 3 examples, 0 constants
        num_examples = 1
        num_genes = 2
        num_features = 3
        targets = [0] * num_examples

        # create
        constants = []
        eval_matrix = np.zeros((num_genes, num_examples))
        data_matrix = np.zeros((num_examples, num_features))

        # simulate the use of a constant in the other, first gene,
        eval_matrix[0, 0] = 2

        # expected; first gene is unchanged; the 2nd one is the sum of the first with itself (i.e. 4)
        expected_eval_matrix = np.array([[2], [4]])

        # run the evaluate
        error = gene.evaluate(1, eval_matrix, data_matrix, constants, targets)
        self.assertTrue(np.array_equal(expected_eval_matrix, eval_matrix))
