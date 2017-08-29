import sys
import datetime as dt
import json
import logging
import os
from dataset import DataSet
from mep.genetics.population import Population

if __name__ == "__main__":
    # TODO: error check usage

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

    # construct a population and run it for the number of generations specified
    population = Population(data_set.data_matrix, data_set.target, int(config["num_constants"]),
                            float(config["constants_min"]), float(config["constants_max"]),
                            float(config["feature_variables_probability"]),
                            int(config["code_length"]), int(config["population_size"]),
                            float(config["operators_probability"]))
    population.initialize()

    # iterate through the generations
    best_chromosome = None
    for generation in range(int(config["num_generations"])):
        best_chromosome = population.chromosomes[0]
        logger.debug("Generation number {} best chromosome error {} with {} chromosomes ".format(
            generation, best_chromosome.error, len(population.chromosomes)))

        if best_chromosome.error == 0:
            logger.debug("Exiting early as we have hit the best possible error.")
            break
        population.next_generation()

    logger.debug("Best chromosome error {} and chromosome (pretty)\n {}".format(best_chromosome.error,
                                                                                best_chromosome.pretty_string()))

    # TODO: this should probably be optional
    # prune out the unused genes
    best_chromosome.prune()
    logger.debug("Best chromosome error {} and chromosome (pretty)\n {}".format(best_chromosome.error,
                                                                                best_chromosome.pretty_string()))

    # TODO: Optional?
    # we then convert the chromosome into a valid python program and write it out to file
    with open(python_file_name, 'w') as python_file:
        python_program = best_chromosome.to_python()
        logger.debug("Write out the python program to {}".format(python_file_name))
        logger.debug(python_program)
        python_file.write(python_program)

    # TODO: Convert the output to a valid python program
    # TODO: Add support for classification
    # TODO: Add example digital circuit test
    # TODO: Add UDFs
