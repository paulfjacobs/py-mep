from abc import ABCMeta, abstractmethod


# TODO: add some more interesting operators; example pow(...), log(...), exp(...)
class Operator(metaclass=ABCMeta):
    """
    This is more of a function than a traditional "operator" but the function could be simply using an operator
    like "+", "-", etc. At it's core these are indivisible functions that take arguments (i.e. preceding genes) and
    output some value.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Run the operation/function and return the result.
        """

    @abstractmethod
    def function_name(self):
        """
        Return the name of the function for use in the pretty print and the python program.
        """

    @abstractmethod
    def function_python_definition(self):
        """
        Return the python definition of the function
        """

    def __str__(self):
        return self.function_name()

    def __repr__(self):
        return str(self)


# TODO: Consolidate these into just one Operator?
class AdditionOperator(Operator):
    """
    Perform addition.
    """
    # NOTE: there is no need to support more than two arguments as these can be chained across multiple genes

    def __call__(self, *args, **kwargs):
        """
        Perform addition.
        """
        return sum(args)

    def function_name(self):
        return "add"

    def function_python_definition(self):
        return """
def add(x, y):
    return x + y
        """


class MultiplicationOperator(Operator):
    """
    Perform multiplication
    """
    # NOTE: there is no need to support more than two arguments as these can be chained across multiple genes

    def __call__(self, *args, **kwargs):
        """
        Perform subtraction.
        """
        result = 1
        for arg in args:
            result *= arg

        return result

    def function_name(self):
        return "multiplication"

    def function_python_definition(self):
        return """
def multiplication(x, y):
    return x * y
        """


class SubtractionOperator(Operator):
    """
    Perform subtraction.
    """
    # NOTE: there is no need to support more than two arguments as these can be chained across multiple genes

    def __call__(self, *args, **kwargs):
        """
        Perform subtraction.
        """
        result = args[0]
        for arg in args[1:]:
            result -= arg

        return result

    def function_name(self):
        return "subtraction"

    def function_python_definition(self):
        return """
def subtraction(x, y):
    return x - y
        """


class MinOperator(Operator):
    """
    Perform the Min operation.
    """

    def __call__(self, *args, **kwargs):
        """
        Perform min
        """
        return min(args)

    def function_name(self):
        return "min_"

    def function_python_definition(self):
        return """
def min_(x, y):
    return min(x, y)
        """


class MaxOperator(Operator):
    """
    Perform the Max operation.
    """

    def __call__(self, *args, **kwargs):
        """
        Perform max
        """
        return max(args)

    def function_name(self):
        return "max_"

    def function_python_definition(self):
        return """
def max_(x, y):
    return max(x, y)
        """