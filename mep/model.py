import logging
from mep.genetics.population import Population


# NOTE: The idea is to explicitly conform to a scikit-learn type of approach where we can run fit(..) and
# predict(..) methods on the model
class MEPModel(object):
    """
    Encapsulate the MEP model.
    """

    def __init__(self, num_constants, constants_min, constants_max, feature_variables_probability, code_length,
                 population_size, operators_probability, num_generations):

        """
        Initialize.
        :param num_constants:
        :param constants_min:
        :param constants_max:
        :param feature_variables_probability:
        :param code_length:
        :param population_size:
        :param operators_probability:
        :param num_generations:
        """
        # logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # core parameters
        self.num_constants = num_constants
        self.constants_min = constants_min
        self.constants_max = constants_max
        self.feature_variables_probability = feature_variables_probability
        self.code_length = code_length
        self.population_size = population_size
        self.operators_probability = operators_probability
        self.num_generations = num_generations

        # the best found chromosome from the evolution process
        self.best_chromosome = None

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        # construct a population and run it for the number of generations specified
        population = Population(X, y, self.num_constants,
                                self.constants_min, self.constants_max,
                                self.feature_variables_probability,
                                self.code_length, self.population_size,
                                self.operators_probability)
        population.initialize()

        # iterate through the generations
        for generation in range(self.num_generations):
            self.best_chromosome = population.chromosomes[0]
            self.logger.debug("Generation number {} best chromosome error {} with {} chromosomes ".format(
                generation, self.best_chromosome.error, len(population.chromosomes)))

            if self.best_chromosome.error == 0:
                self.logger.debug("Exiting early as we have hit the best possible error.")
                break
            population.next_generation()

        self.logger.debug("Best chromosome error {} and chromosome (pretty)\n {}".format(
            self.best_chromosome.error, self.best_chromosome.pretty_string()))

        # prune out the unused genes
        self.best_chromosome.prune()

    def predict(self, X):
        """
        Return the predictions for this data.
        :param X: the sample data; matrix with (n_samples, n_features)
        :type X: np.matrix
        :return: the prediction for each sample; array-like (n_samples) length
        :rtype: np.array
        """
        return self.best_chromosome.predict(X)

    def score(self, X, y):
        """
        Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
        ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y, disregarding the input features, would get a
        R^2 score of 0.0.

        (NOTE: Comment taken from scikit-learn.)
        :param X: the sample data; matrix with (n_samples, n_features)
        :type X: np.matrix
        :param y: the target values
        :type y: array-like, shape = (n_samples)
        :return: the score
        :rtype: float
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()

        return 1 - u/v

    # NOTE: These are NOT scikit-learn methods now
    def to_python(self):
        """
        Return a python program which can run the model directly via direct inputs.
        :return: the python program (string)
        :rtype: str
        """
        if self.best_chromosome is None:
            raise ValueError("The model hasn't been fit.")

        return self.best_chromosome.to_python()
