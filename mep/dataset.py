import pandas as pd


class DataSet:
    """
    Encapsulate a data set. Feature vectors and their targets.
    """

    def __init__(self, filename):
        """
        Initialize.

        :param filename: the filename (full path to CSV) of the data
        :type filename: str
        """
        # we assume this in the format of feature cols and then target
        self.data = pd.read_csv(filename)

        # extract out data matrix and target
        self.target = self.data.target.values
        self.data_matrix = self.data.drop("target", 1).values
