import torch
import numpy as np
from utils import train_load,dev_load
from torch.utils.data import Dataset
import torch.utils.data as utils
import torch.nn as nn
from torchvision.models import resnet18
from model import SpeechModel, BasicBlock
from torchsummary import summary
import os
from torch.autograd import Variable

def main():
    K = 5000
    batch_size = 32
    embedding_size = 1024
    learning_rate = 0.001
    nepoch = 30
    weight_decay = 1e-3

    class CustomerDataset(Dataset):
        def __init__(self, dataset, K):
            self.K = K
            features, speakers = dataset
            self.features = torch.from_numpy(np.asarray(features))
            self.speakers = torch.from_numpy(np.asarray(speakers))

        def __len__(self):
            return self.speakers.shape[0]

        def __getitem__(self, index):
            feature = self.features[index]
            speaker = self.speakers[index]
            if feature.shape[0] <= self.K:
                upper = (self.K - feature.shape[0]) // 2
                lower = self.K - feature.shape[0] - upper
                feature = np.pad(feature, ((upper, lower), (0,0)), 'reflect')
            else:
                start = np.random.randint(0, self.K - feature.shape[0])
                feature = feature[start:start+self.K]
            return (feature, speaker)

    class TestDataset(Dataset):
        def __init__(self, dataset, K):
            self.K = K
            trials, label, enrol, test = Dataset
            self.enrol = torch.from_numpy(np.asarray(enrol))
            self.test = torch.from_numpy(np.asarray(test))
            self.trials = np.asarray(trials)
            self.label = np.asarray(label)

        def __len__(self):
            return self.label.shape[0]

        def get_K(self, feature):
            if feature.shape[0] <= self.K:
                upper = (self.K - feature.shape[0]) // 2
                lower = self.K - feature.shape[0] - upper
                feature = np.pad(feature, ((upper, lower), (0,0)), 'reflect')
            else:
                start = np.random.randint(0, self.K - feature.shape[0])
                feature = feature[start:start+self.K]
            return feature

        def __getitem__(self, index):
            feature1 = self.enrol[index][0]
            feature2 = self.test[index][1]
            feature1 = self.get_K(feature1)
            feature2 = self.get_K(feature2)
            if self.label is None:
                label = None
            else:
                label = self.label[index]
            return (feature1, feature2, label)

    print("Creating Training Datasets")
    train_features, train_speakers, nspeakers = train_load("./hw2p2_A",[1])
    trainDataset = CustomerDataset([train_features, train_speakers], K)
    trainDataloader = utils.Dataloader(dataset = trainDataset, batch_size = batch_size,
                                       shuffle = True)
    print("Training Datasets Creating Done")

    print("Creating Dev Datasets")
    trials, label, enrol, test  = dev_load("./hw2p2_B/dev.preprocessed.npz")
    DevDataset = CustomerDataset([trials, label, enrol, test], K)
    DevDataloader = utils.Dataloader(dataset = DevDataset, batch_size = batch_size,
                                     shuffle = False)
    print("Validation Datasets Creating Done")

    net = SpeechModel(BasicBlock, [2,2,2,2], embedding_size, 1000)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optimizer.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    def test_on_train(nepoch):
        net.eval()
        correct = 0
        total = 0
        for feature, speaker in trainDataloader:
            feature = Variable(feature).cuda()
            speaker = Variable(speaker).cuda()
            outputs = net(feature)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cuda()
            total += speaker.size(0)
            labels = speaker.view(speaker.shape[0])
            correct += int((predicted.cpu() == labels.cpu()).sum())
        print('Accuracy of the network on the train frames: %.2f %% on %d epoch' % (100 * correct / total, nepoch))
        return correct / total

    def train(nepoch):
        net.train()
        for epoch in range(nepoch):
            for i, (feature, speaker) in enumerate(trainDataloader):
                feature = Variable(feature).cuda()
                speaker = Variable(speaker).cuda()
                optimizer.zero_grad()
                outputs = net(feature)
                loss = criterion(outputs, speaker)
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                          % (epoch + 1, nepoch, i + 1, len(trainDataset) // batch_size, loss.data[0]))

            if (epoch + 1) % 5 == 0:
                test_on_train(epoch)
    train(nepoch)


if __name__ == '__main__':
    main()