ENGG 5189 EC Project README file 

# PACKAGES
	we use tensorflow to build up our system, so please make sure you install the following packages
		- tensorflow (>1.1.0)
		- sugartensor # high level tensorflow wrapper interface 
		- numpy 

# DATASET
	we use the well-known MNIST dataset for training and testing, the program will automatically download for you and save to './asset/data/mnist'

# TRAIN BASELINE MODEL
	python main.py [-dir=YOUR_PATH]

# TRAIN MODEL WITH GP
	python main.py -p train_gp --batch_size=1 [-dir=YOUR_PATH] # better to have different path of the baseline model

# TEST 
	python main.py -p test --batch_size=1 [-dir=YOUR_PATH][--addnoise -v=SELECT_VALUE][--rotate]

# TENSORBOARD
	tensorboard --logdir=YOUR_BASELINE_MODEL_FOLDER # we record the log information of the baseline model
	