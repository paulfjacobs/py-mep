import unittest
from mep.genetics.operator import MultiplicationOperator, AdditionOperator, SubtractionOperator
from mep.genetics.operator import MinOperator, MaxOperator


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

    def test_min_operator(self):
                    """
                    """
                    # construct the operator and test it
                    operator = MinOperator()
                    self.assertEquals(min(5, 2), operator(5, 2))
                    self.assertEquals("min_", operator.function_name())
                    self.assertEquals("""
def min_(x, y):
    return min(x, y)
        """, operator.function_python_definition())

    def test_max_operator(self):
        """
        """
        # construct the operator and test it
        operator = MaxOperator()
        self.assertEquals(max(5, 2), operator(5, 2))
        self.assertEquals("max_", operator.function_name())
        self.assertEquals("""
def max_(x, y):
    return max(x, y)
        """, operator.function_python_definition())