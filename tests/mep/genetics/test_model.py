import unittest
from mep.model import MEPModel
import random
import numpy as np

# make reproducible
random.seed(1)


class TestModel(unittest.TestCase):
    """
    Test the model.
    """

    def test_model_basic(self):
        model = MEPModel(num_constants=2, constants_min=-1, constants_max=1,
                         feature_variables_probability=0.4, code_length=50,
                         population_size=100, operators_probability=0.5,
                         num_generations=200)

        # generate data from this function
        def function_to_learn(x1, x2):
            return x1 + x2
        training_feature_matrix = []
        training_target_vector = []
        for sample in range(100):
            x1, x2 = random.randint(-100, 100), random.randint(-100, 100)
            val = function_to_learn(x1, x2)
            training_feature_matrix.append([x1, x2])
            training_target_vector.append(val)

        # fit the model
        model.fit(np.matrix(training_feature_matrix), np.array(training_target_vector))

        # test data
        def function_to_learn(x1, x2):
            return x1 + x2
        test_feature_matrix = []
        test_target_vector = []
        for sample in range(100):
            x1, x2 = random.randint(-100, 100), random.randint(-100, 100)
            val = function_to_learn(x1, x2)
            test_feature_matrix.append([x1, x2])
            test_target_vector.append(val)

        self.assertEquals(model.score(np.matrix(training_feature_matrix), np.array(training_target_vector)), 1)

