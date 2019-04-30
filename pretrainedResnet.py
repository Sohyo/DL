
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import CIFAR100
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
#from tqdm.autonotebook import tqdm
import argparse


class OurResNet:
    def __init__(self, epochs, train_batch_size=100, val_batch_size=100, num_classes=1000, pretrained=True):
        #load the model
        self.model = models.resnet18(pretrained=pretrained, num_classes=num_classes)
        if pretrained:
            self.model.fc = nn.Linear(512, num_classes)
            
        #params you need to specify:
        self.epochs = epochs

        # put your data loader here
        self.train_loader, self.val_loader = self.get_data_loaders(train_batch_size, val_batch_size)
        self.loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

        # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
        self.optimizer = optim.Adadelta(self.model.parameters())

        # See if we use CPU or GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()


    @staticmethod
    def get_data_loaders(train_batch_size, val_batch_size):
        # Transform function first Resize -> toTensor -> then normalize pixel values
        data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
            
        train_loader = DataLoader(CIFAR100(download=True, root="./data", transform=data_transform, train=True),
                                batch_size=train_batch_size, shuffle=True)

        val_loader = DataLoader(CIFAR100(download=True, root="./data", transform=data_transform, train=False),
                                batch_size=val_batch_size, shuffle=False)
        return train_loader, val_loader
    
    def train(self):
        total_loss = 0

        self.model.train()
        if self.cuda_available:
            self.model.cuda()
        for i, data in enumerate(self.train_loader):
            X, y = data[0].to(self.device), data[1].to(self.device)
            # training step for single batch
            self.model.zero_grad()
            outputs = self.model(X)

            loss = self.loss_function(outputs, y)
            loss.backward()
            self.optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            #progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
            if not self.cuda_available:
                print(total_loss/(i+1))
            
        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss
        
    def validate(self):
        
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                X, y = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(X) # this get's the prediction from the network

                val_losses += self.loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
                
                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy), 
                                    (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        self.calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )
        return val_losses, precision, recall, f1, accuracy
    
    def run(self):
        start_ts = time.time()
        
        losses = []
        batches = len(self.train_loader)
        val_batches = len(self.val_loader)
        print("batchs: {}, val_batches: {}".format(batches, val_batches))
        
        for epoch in range(self.epochs):
            total_loss = self.train()
            val_losses, precision, recall, f1, accuracy = self.validate()
            
            print(f"Epoch {epoch+1}/{self.epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
            self.print_scores(precision, recall, f1, accuracy, val_batches)
            losses.append(total_loss/batches) # for plotting learning curve
        
        print(f"Training time: {time.time()-start_ts}s")
    
    @staticmethod
    def calculate_metric(metric_fn, true_y, pred_y):
        # multi class problems need to have averaging method
        if "average" in inspect.getfullargspec(metric_fn).args:
            return metric_fn(true_y, pred_y, average="macro")
        else:
            return metric_fn(true_y, pred_y)
    
    @staticmethod
    def print_scores(p, r, f1, a, batch_size):
        # just an utility printing function
        for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
            print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--filename', type=str, default='default',
                        help='The nice file name to store nice output.')

    args = parser.parse_args()
    
    return args.epochs, args.filename


if __name__ == '__main__':
    epochs, filename = parse_arguments()
    print("Our filename: {}".format(filename))
    res = OurResNet(epochs=epochs)
    res.run()