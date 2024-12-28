import logging


def get_logger(file_name, data_name, data_rate):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    filename = "./1.logs/" + data_name + data_rate + ".log"
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
