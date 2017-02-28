import pandas as pd


class DataSet(object):
    """
    Encapsulate a data set. Feature vectors and their targets.
    """

    def __init__(self, filename):
        """
        Initialize.

        :param filename: the filename (full path to CSV) of the data
        :type filename: str
        """
        # TODO: What about supporting other file formats?
        self.data = pd.read_csv(filename)