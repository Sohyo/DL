import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Averages and plots output fom our pytorch networks.')
    parser.add_argument('--amount_of_files', type=int, default=4,
                        help='The amount of runs we did.')
    parser.add_argument('--filename', type=str, default='default',
                        help='The base filename of the files we want to read in.')
    parser.add_argument('--everything', default=False, action='store_true')
    

    args = parser.parse_args()

    return args.amount_of_files, args.filename, args.everything


def read_in_files_to_average(filename, amount_of_files):
	summed_data = None
	for file_number in range(amount_of_files):
		with open("{}_{}".format(filename, file_number), 'r') as f:
			data = np.array(list(csv.reader(f))[1:]).astype(float)
			if summed_data is None:
				summed_data = np.zeros(data.shape)
			summed_data += data
	return summed_data / amount_of_files

def plot_accuracy(averages):
	plt.plot(averages[:,5])
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.show()

def plot_losses(averages):
	plt.plot(averages[:,0])
	plt.plot(averages[:,1])
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.show()

def plot_multiple_accuracies(list_of_averages, filenames):
	for averages in list_of_averages:
		plt.plot(averages[:,5])
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.legend(filenames, loc='bottom right')
	plt.show()


def plot_everything():
	filenames = ["{}_{}_400_{}".format(net, method, optimizer) for net in ['dense', 'res'] for method in ['fe', 'ft'] for optimizer in ['adam', 'adadelta']]
	all_averages = [read_in_files_to_average(filename, 4) for filename in filenames]
	plot_multiple_accuracies(all_averages, filenames)

if __name__ == '__main__':
    amount_of_files, filename, everything = parse_arguments()
    if everything:
        plot_everything()
    else:
        averages = read_in_files_to_average(filename, amount_of_files)
        plot_accuracy(averages)
        plot_losses(averages)
	
#training_loss,validation_loss,precision,recall,f1,accuracy
