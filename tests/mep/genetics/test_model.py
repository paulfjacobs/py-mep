import unittest
from mep.model import MEPModel
import random
import numpy as np
import logging
import datetime as dt

# make reproducible
random.seed(1)

logging.basicConfig(filename="output_logs/TEST_{}.log".format(dt.datetime.now().strftime("%Y%m%d")),
                    level=logging.DEBUG,
                    filemode='w',
                    format="%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s")
logger = logging.getLogger("main")


class TestModel(unittest.TestCase):
    """
    Test the model.
    """

    def _generate_train_and_test(self, function_to_learn, num_samples, num_args):
        training_feature_matrix = []
        training_target_vector = []
        for sample in range(num_samples):
            args = [random.randint(-250, 250) for _ in range(num_args)]
            training_feature_matrix.append(args)
            training_target_vector.append(function_to_learn(*args))

        test_feature_matrix = []
        test_target_vector = []
        for sample in range(num_samples):
            args = [random.randint(-250, 250) for _ in range(num_args)]
            test_feature_matrix.append(args)
            test_target_vector.append(function_to_learn(*args))

        return np.matrix(training_feature_matrix), np.array(training_target_vector), \
               np.matrix(test_feature_matrix), np.array(test_target_vector)

    def test_model_basic(self):
        model = MEPModel(num_constants=2, constants_min=-1, constants_max=1,
                         feature_variables_probability=0.4, code_length=50,
                         population_size=100, operators_probability=0.5,
                         num_generations=200)

        # generate data from this function
        def function_to_learn(x1, x2):
            return x1 + x2

        training_feature_matrix, training_target_vector, test_feature_matrix, test_target_vector = self._generate_train_and_test(
            function_to_learn, 100, 2)

        # fit the model
        model.fit(training_feature_matrix, training_target_vector)

        self.assertEqual(model.score(test_feature_matrix, test_target_vector), 1)

    def test_model_min_max(self):
        model = MEPModel(num_constants=2, constants_min=-1, constants_max=1,
                         feature_variables_probability=0.4, code_length=50,
                         population_size=100, operators_probability=0.7,
                         num_generations=200)

        # generate data from this function
        def function_to_learn(x1, x2, x3, x4):
            return min(x1, x2) + max(x3, x4)

        training_feature_matrix, training_target_vector, test_feature_matrix, test_target_vector = self._generate_train_and_test(
            function_to_learn, 100, 4)

        # fit the model
        model.fit(training_feature_matrix, training_target_vector)

        self.assertEqual(model.score(test_feature_matrix, test_target_vector), 1)

    def test_model_pow(self):
        model = MEPModel(num_constants=2, constants_min=-1, constants_max=1,
                         feature_variables_probability=0.4, code_length=50,
                         population_size=100, operators_probability=0.5,
                         num_generations=200)

        # generate data from this function
        def function_to_learn(x1, x2, x3, x4):
            return x1 * x2 + x2 * x2 + x3

        training_feature_matrix, training_target_vector, test_feature_matrix, test_target_vector = self._generate_train_and_test(
            function_to_learn, 100, 4)

        # fit the model
        model.fit(training_feature_matrix, training_target_vector)

        self.assertEqual(model.score(test_feature_matrix, test_target_vector), 1)
