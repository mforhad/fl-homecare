import yaml

class FlHomecareConfig:
    def __init__(self) -> None:
        self.config = self.load_config()
    
    def load_config(self):
        with open("hc-config.yaml", 'r') as yamlfile:
            self.config = yaml.safe_load(yamlfile)
            print("Configuration loaded successfully!")
            print(self.config)
        
        # define public variables for each configuration
        # Strategy configurations
        self.num_clients = self.config["strategy"]["num_clients"]
        self.fraction_fit = self.config["strategy"]["fraction_fit"]
        self.fraction_eval = self.config["strategy"]["fraction_eval"]
        self.min_fit_clients = self.config["strategy"]["min_fit_clients"]
        self.min_evaluate_clients = self.config["strategy"]["min_evaluate_clients"]

        # Server configurations
        self.num_rounds = self.config["server-config"]["num_rounds"]
        self.round_timeout = self.config["server-config"]["round_timeout"]

        # Model configurations
        self.lr = self.config["model"]["lr"]
        self.num_epochs = self.config["model"]["num_epochs"]
        self.batch_size = self.config["model"]["batch_size"]

        # Data configurations
        self.val_ratio = self.config["data"]["val_ratio"]
        self.is_data_from_s3 = self.config["data"]["data_from_s3"]
        self.is_heterogeneous = self.config["data"]["heterogeneous_data_set"]

        # Metrics Configurations
        self.energy_weight = self.config["metrics"]["energy_weight"]
        self.accuracy_weight = self.config["metrics"]["accuracy_weight"]
        self.train_time_weight = self.config["metrics"]["train_time_weight"]

        # Miscellaneous configurations
        self.should_use_gpu = self.config["use_gpu"]


    