import unittest
from mep.genetics.operator import MultiplicationOperator, AdditionOperator, SubtractionOperator


class TestOperators(unittest.TestCase):
    """
    Test the Operator classes
    """

    def test_multiplication_operator(self):
        """
        """
        # construct the operator and test it
        operator = MultiplicationOperator()
        self.assertEquals(5 * 2, operator(5, 2))
        self.assertEquals("multiplication", operator.function_name())
        self.assertEquals("""
def multiplication(x, y):
    return x * y
        """, operator.function_python_definition())

    def test_addition_operator(self):
            """
            """
            # construct the operator and test it
            operator = AdditionOperator()
            self.assertEquals(5 + 2, operator(5, 2))
            self.assertEquals("add", operator.function_name())
            self.assertEquals("""
def add(x, y):
    return x + y
        """, operator.function_python_definition())

    def test_subtraction_operator(self):
                """
                """
                # construct the operator and test it
                operator = SubtractionOperator()
                self.assertEquals(5 - 2, operator(5, 2))
                self.assertEquals("subtraction", operator.function_name())
                self.assertEquals("""
def subtraction(x, y):
    return x - y
        """, operator.function_python_definition())