import argparse
import sys
import datetime as dt
import json
import logging
import os
from mep.dataset import DataSet
from mep.model import MEPModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the MEP model.\nExample: 'python -m mep.main datasets/data1.csv test.py", allow_abbrev=False)
    parser.add_argument("data_set_name", help="The name (full path) to the data file to train on.")
    parser.add_argument("python_file_name", help="The name (full path) to the python file to write the output program.")
    args = parser.parse_args()

    # get the data file
    data_set_name = args.data_set_name
    python_file_name = args.python_file_name
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

    # we then convert the chromosome into a valid python program and write it out to file
    if python_file_name:
        python_program = model.to_python()
        with open(python_file_name, 'w') as python_file:
            logger.debug("Write out the python program to {}".format(python_file_name))
            logger.debug(python_program)
            python_file.write(python_program)

    # TODO: Add support for classification
    # TODO: Add example digital circuit test
    # TODO: Add UDFs
