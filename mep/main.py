import sys
import datetime as dt
import json
import logging
import os
from dataset import DataSet
from mep.model import MEPModel

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Expected usage 'python -m mep.main DATA_SET_NAME PYTHON_FILE_NAME'\n" +
              "     DATA_SET_NAME:    The name (full path) to the data file to train on.\n"
              "     PYTHON_FILE_NAME: The name (full path) to the python file to write the output program.\n"
              "Example: 'python -m mep.main datasets/data1.csv test.py'"
              )
        sys.exit(-1)

    # get the data file
    data_set_name = sys.argv[1]
    python_file_name = sys.argv[2]
    data_set = DataSet(data_set_name)

    # read config file
    # TODO: Possible config file override on command line
    with open("mep/config/config.json") as config_file:
        config = json.load(config_file)

    # construct output logs dir if it doesn't exist
    output_logs_dir = config["output_logs"]
    if not os.path.exists(output_logs_dir):
        os.mkdir(output_logs_dir)

    # configure logger
    logging.basicConfig(filename="{}/MEP_{}.log".format(output_logs_dir, dt.datetime.now().strftime("%Y%m%d")),
                        level=logging.DEBUG,
                        filemode='w',
                        format="%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s")
    logger = logging.getLogger("main")
    logger.info("Starting up...")

    # configure the model; then fit it to the training data
    model = MEPModel(int(config["num_constants"]), float(config["constants_min"]), float(config["constants_max"]),
                     float(config["feature_variables_probability"]), int(config["code_length"]),
                     int(config["population_size"]), float(config["operators_probability"]),
                     int(config["num_generations"]))
    model.fit(data_set.data_matrix, data_set.target)
    logger.info("Finished fitting the model")

    # TODO: Optional?
    # we then convert the chromosome into a valid python program and write it out to file
    with open(python_file_name, 'w') as python_file:
        python_program = model.to_python()
        logger.debug("Write out the python program to {}".format(python_file_name))
        logger.debug(python_program)
        python_file.write(python_program)

    # TODO: Add support for classification
    # TODO: Add example digital circuit test
    # TODO: Add UDFs
