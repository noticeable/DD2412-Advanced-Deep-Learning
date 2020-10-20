from pytorch_lightning.loggers import CometLogger
import toml

def create_logger(experiment_name):
    COMET = toml.load('config.toml')['comet']
    print('Running experiment', experiment_name)
    logger = CometLogger(api_key=COMET["api_key"], project_name=COMET["project_name"], experiment_name=experiment_name)
    return logger