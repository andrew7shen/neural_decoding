# Script for background processes

# Import packages
import sys
import yaml

class Config():

    def __init__(self, args):
        self.T = args["T"]
        self.N = args["N"]
        self.M = args["M"]
        self.d = args["d"]  # num_modes
        self.b = args["b"]
        self.type = args["type"]
        self.epochs = args["epochs"]
        self.lr = args["lr"]
        self.record = args["record"]
        self.ev = args["ev"]
        self.m1_path = args["m1_path"]
        self.emg_path = args["emg_path"]
        self.behavioral_path = args["behavioral_path"]
        self.save_path = args["save_path"]
        self.model = args["model"]
        self.seed = args["seed"]
        self.temperature = args["temperature"]
        self.weight_decay = args["weight_decay"]
        self.hidden_dim = args["hidden_dim"]
        self.cluster_model_type = args["cluster_model_type"]
        self.decoder_model_type = args["decoder_model_type"]
        self.kmeans_cluster = args["kmeans_cluster"]
        self.set_decoder_weights = args["set_decoder_weights"]
        self.label_type = args["label_type"]
        self.remove_zeros = args["remove_zeros"]
        self.scale_outputs = args["scale_outputs"]
        self.run_id = args["run_id"]
        self.anneal_temperature = args["anneal_temperature"]
        self.end_temperature = args["end_temperature"]


def load_config():
    """
    Loads configuration file and reads values for usage.
    Input: None
    Output: (Config) config object
    """

    try:
        config_path = sys.argv[1]
    except Exception:
        raise Exception("Must include config file to run.")

    args = yaml.safe_load(open(config_path, "r"))

    return Config(args)

