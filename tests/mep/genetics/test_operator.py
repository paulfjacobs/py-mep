import unittest
from mep.genetics.operator import MultiplicationOperator


class TestOperators(unittest.TestCase):
    """
    Test the Operator classes
    """

    def test_multiplication_operator(self):
        """
        """
        # construct the oeprator
        operator = MultiplicationOperator()
        self.assertEquals(5 * 2, operator(5, 2))