import yaml

class FlHomecareConfig:
    def __init__(self) -> None:
        self.config = self.load_config()
    
    def load_config(self):
        with open("fl-config.yaml", 'r') as yamlfile:
            self.config = yaml.safe_load(yamlfile)
        
        # define public variables for each configuration
        self.num_epochs = self.config["num_epochs"]
        self.num_clients = self.config["num_clients"]
        self.num_rounds = self.config["num_rounds"]
        self.lr = self.config["lr"]
        self.data_from_s3 = self.config["data_from_s3"]
        self.batch_size = self.config["batch_size"]
        self.val_ratio = self.config["val_ratio"]

    
