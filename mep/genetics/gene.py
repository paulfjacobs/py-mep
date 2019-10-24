import logging
import numpy as np
from abc import ABCMeta, abstractmethod


class Gene(metaclass=ABCMeta):
    """
    Lowest level of the genetic structure of MEP. Think of this as one line of code in the program.
    """

    @abstractmethod
    def evaluate(self, gene_index, eval_matrix, data_matrix, constants, targets):
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
        :param targets: the targets; equal to the number of examples (n)
        :type targets: list
        :return: error (sum of error across the examples); modifies the eval_matrix
        :rtype: float
        """


# NOTE: Should we also add a mutate method to the gene itself? Considering that we are doing the mutation by doing
# a crossover of the whole chromosome with a new random chromosome, I don't think there is any benefit.


class VariableGene(Gene):
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
        # self.logger = logging.getLogger(self.__class__)

        self.index = index
        self.is_feature = is_feature

    def evaluate(self, gene_index, eval_matrix, data_matrix, constants, targets):
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
        :param targets: the targets; equal to the number of examples (n)
        :type targets: list
        :return: error (sum of error); modifies the eval_matrix
        :rtype: float
        """
        # TODO: Move common logic up
        # TODO: Handle classification as well as regression

        # go through and set the data
        num_examples = eval_matrix.shape[1]
        sum_of_errors = 0.
        for example_index in range(0, num_examples):
            # each column is one example in the data matrix (i.e. one feature vector)
            # if we are a feature variable then we look at the corresponding feature in the feature vector for this
            # example; otherwise (as a constant) we just go to that (independent of the example we are in)
            if self.is_feature:
                value = data_matrix[example_index, self.index]
            else:
                value = constants[self.index]
            # calculate error
            sum_of_errors += abs(targets[example_index] - value)

            # set it in the eval matrix
            eval_matrix[gene_index, example_index] = value

        return sum_of_errors

    def __str__(self):
        return "VariableGene({}, is_feature={})".format(self.index, self.is_feature)

    def pretty_string(self):
        """
        Pretty program string version.
        :return: string version
        :rtype: str
        """
        if self.is_feature:
            return "FEATURES[{}]".format(self.index)
        else:
            return "CONSTANTS[{}]".format(self.index)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other is None or not isinstance(other, VariableGene):
            return False
        return self.index == other.index and self.is_feature == other.is_feature


class OperatorGene(Gene):
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
        # self.logger = logging.getLogger(self.__class__)

        self.operation = operation
        self.address1 = address1
        self.address2 = address2

    def evaluate(self, gene_index, eval_matrix, data_matrix, constants, targets):
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
        :param targets: the targets; equal to the number of examples (n)
        :type targets: list
        :return: error (sum of error); modifies the eval_matrix
        :rtype: float

        """
        # go through and set the data
        num_examples = eval_matrix.shape[1]
        sum_of_errors = 0.
        for example_index in range(0, num_examples):
            # each column is one example in the data matrix (i.e. one feature vector)

            # TODO: Catch errors; in particular division can be a problem
            value = self.operation(eval_matrix[self.address1][example_index],
                                   eval_matrix[self.address2][example_index])
            # set it in the eval matrix
            eval_matrix[gene_index, example_index] = value

            sum_of_errors += abs(targets[example_index] - value)

        return sum_of_errors

    def __str__(self):
        return "OperatorGene({}, {}, {})".format(self.operation, self.address1, self.address2)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other is None or not isinstance(other, OperatorGene):
            return False

        # NOTE: the operators are the same if they are of the same type
        return isinstance(self.operation, type(other.operation)) and self.address1 == other.address1 and self.address2 == other.address2
