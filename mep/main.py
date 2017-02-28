import sys
import datetime as dt
import json
import logging
import os

if __name__ == "__main__":
    # TODO: Get the data file

    # read config file
    # TODO: Possible config file override on comand line
    with open("mep/config/config.json") as data_file:
        config = json.load(data_file)

    # construct output logs dir if it doesn't exist
    output_logs_dir = config["output_logs"]
    if not os.path.exists(output_logs_dir):
        os.mkdir(output_logs_dir)

    # configure logger
    logging.basicConfig(filename="{}/MEP_{}.log".format(output_logs_dir, dt.datetime.now().strftime("%Y%m%d")),
                        level=logging.DEBUG,
                        filemode='w',
                        format="%(asctime)s %(name)s %(funcName)s %(levelname) %(message)s")
    logger = logging.getLogger("main")
    logger.info("Starting up...")

    

