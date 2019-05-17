# DL

datasets.py
This file loads the npy files in the data folder to a pytorch dataset, which can be used in a DataLoader.

networks.py
This file implements a BaseNet class that handles the initialization, training and validation of networks.
It also contains two subclasses of this class that are adjusted for a Resnet and a Densenet.

main.py
This file imports the dataset from datasets.py and the networks from networks.py.
It handles some command line arguments parsing, trains the chosen network and saves the metrics to a file.
Run python main.py --help to see an explanation of the possible arguments.

plotter.py
This file was used to create the plots of the report.
