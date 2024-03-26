import yaml

class FlHomecareConfig:
    def __init__(self) -> None:
        self.config = self.load_config()
    
    def load_config(self):
        with open("fl-config.yaml", 'r') as yamlfile:
            self.config = yaml.safe_load(yamlfile)
            print("Configuration loaded successfully!")
            print(self.config)
        
        # define public variables for each configuration
        self.num_epochs = self.config["num_epochs"]
        self.num_clients = self.config["num_clients"]
        self.num_rounds = self.config["num_rounds"]
        self.lr = self.config["lr"]
        self.is_data_from_s3 = self.config["data_from_s3"]
        self.batch_size = self.config["batch_size"]
        self.val_ratio = self.config["val_ratio"]
        self.is_heterogeneous = self.config["heterogeneous_data_set"]
        self.should_use_gpu = self.config["use_gpu"]
        self.fraction_fit = self.config["fraction_fit"]
        self.fraction_eval = self.config["fraction_eval"]
        self.min_fit_clients = self.config["min_fit_clients"]
        self.min_evaluate_clients = self.config["min_evaluate_clients"]
        self.round_timeout = self.config["round_timeout"]

    
