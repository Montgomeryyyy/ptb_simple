import os


def get_config_path():
    return os.getenv("PTB_SIMPLE_CONFIGS")


def get_data_path():
    return os.getenv("PTB_SIMPLE_DATA")
