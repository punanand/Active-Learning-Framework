from oracle import Oracle
from dataset import Config, Dataset

if __name__ == "__main__":
	config = Config(10, 10)
	data = Dataset(config)
	orac = Oracle(data, config)