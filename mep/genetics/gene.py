import logging
import numpy as np
from abc import ABCMeta, abstractmethod


class Gene(object):
    """
    Lowest level of the genetic structure of MEP. Think of this as one line of code in the program.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, gene_index, eval_matrix, data_matrix, constants):
        """
        This method will modify the eval_matrix for this gene index for each example in the data_matrix.

        :param gene_index: the row index for this gene in the eval_matrix.
        :type gene_index: int
        :param eval_matrix: this is a (c, n) matrix where 'c' is the number of genes in the chromosome and 'n' is the
        number of examples in the training data. The valuate at eval_matrix[i, j] then is the value of gene "i"
        evaluated with the data example (i.e. feature vector) at row "j" in the data matrix.
        :type eval_matrix: np.matrix
        :param data_matrix: the data matrix; rows are feature vectors; comes from the data set
        :type data_matrix: np.matrix
        :param constants: the constants associated with this chromosome
        :type constants: list
        :return: nothing; modifies the eval_matrix
        """
# TODO: Should we also add a mutate method to the gene itself?


class VariableGene(object):
    """
    This gene is simply a variable. Either a constant or one of the features in the data -- i.e. an input variable.
    """

    def __init__(self, index, is_feature=True):
        """
        The index into either the feature vector (if "is_feature" is True) or into the constants.
        :param index: the index into the vector
        :type index: int
        :param is_feature: whether this is a feature variable or a constant
        :type is_feature: bool
        """
        self.logger = logging.getLogger(self.__class__)

        self.index = index
        self.is_feature = is_feature

    def evaluate(self, gene_index, eval_matrix, data_matrix, constants):
        """
        This method will modify the eval_matrix for this gene index for each example in the data_matrix.

        :param gene_index: the row index for this gene in the eval_matrix.
        :type gene_index: int
        :param eval_matrix: this is a (c, n) matrix where 'c' is the number of genes in the chromosome and 'n' is the
        number of examples in the training data. The valuate at eval_matrix[i, j] then is the value of gene "i"
        evaluated with the data example (i.e. feature vector) at row "j" in the data matrix.
        :type eval_matrix: np.matrix
        :param data_matrix: the data matrix; rows are feature vectors; comes from the data set; it is (n, m) where "n"
        is the number of examples and "m" is the number of features.
        :type data_matrix: np.matrix
        :param constants: the constants associated with this chromosome
        :type constants: list
        :return: nothing; modifies the eval_matrix
        """
        # go through and set the data
        num_examples = eval_matrix.shape[1]
        for example_index in range(0, num_examples):
            # each column is one example in the data matrix (i.e. one feature vector)
            # if we are a feature variable then we look at the corresponding feature in the feature vector for this
            # example; otherwise (as a constant) we just go to that (independent of the example we are in)
            if self.is_feature:
                eval_matrix[gene_index, example_index] = data_matrix[example_index, self.index]
            else:
                eval_matrix[gene_index, example_index] = constants[self.index]


class OperatorGene(object):
    """
    This gene performance an operation on two addresses. The addresses are indices in the eval_matrix -- i.e. from the
    evaluation of other genes before this one.
    """

    # NOTE: This could be expanded to multiple addresses
    def __init__(self, operation, address1, address2):
        """
        Initialize.
        :param operation: a lambda or function that can be operated on two floats
        :type operation: lambda
        :param address1: index into the eval_matrix
        :type address1: int
        :param address2: index into the eval_matrix
        :type address2: int
        """
        self.logger = logging.getLogger(self.__class__)

        self.operation = operation
        self.address1 = address1
        self.address2 = address2

    def evaluate(self, gene_index, eval_matrix, data_matrix, constants):
        """
        This method will modify the eval_matrix for this gene index for each example in the data_matrix.

        :param gene_index: the row index for this gene in the eval_matrix.
        :type gene_index: int
        :param eval_matrix: this is a (c, n) matrix where 'c' is the number of genes in the chromosome and 'n' is the
        number of examples in the training data. The valuate at eval_matrix[i, j] then is the value of gene "i"
        evaluated with the data example (i.e. feature vector) at row "j" in the data matrix.
        :type eval_matrix: np.matrix
        :param data_matrix: the data matrix; rows are feature vectors; comes from the data set
        :type data_matrix: np.matrix
        :param constants: the constants associated with this chromosome
        :type constants: list
        :return: nothing; modifies the eval_matrix
        """
        # go through and set the data
        num_examples = eval_matrix.shape[1]
        for example_index in range(0, num_examples):
            # each column is one example in the data matrix (i.e. one feature vector)

            # TODO: Catch errors; in particular division can be a problem
            eval_matrix[gene_index, example_index] = self.operation(eval_matrix[self.address1][example_index],
                                                                    eval_matrix[self.address2][example_index])