import copy
from datetime import datetime
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from time import time
import matplotlib.pyplot as plt

SQUEEZENET_INPUT_SIZE = 224

class OwnNet(nn.Module):
    def __init__(self):
        super(OwnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = num_classes
    return model_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_train_loader_own(batch_size_train):
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/files/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_train, shuffle=True, num_workers=2, pin_memory=True)
    return train_loader

def get_test_loader_own(batch_size_test):
    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_test, shuffle=True, num_workers=2, pin_memory=True)
    return test_loader

def expand_img(img):
    return img.expand(3,-1,-1)

def get_train_loader_squeeze(batch_size_train):
    dataset = datasets.MNIST('../data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(SQUEEZENET_INPUT_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Lambda(expand_img)
                                        ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True, num_workers=1, pin_memory=True)
    return train_loader

def get_test_loader_squeeze(batch_size_test):
   dataset = datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                           transforms.Resize(SQUEEZENET_INPUT_SIZE),
                           transforms.ToTensor(),
                           transforms.Lambda(expand_img)
                       ]))
   test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, shuffle=True, num_workers=1, pin_memory=True)
   return test_loader

def train(model, device, dataloader, criterion, optimizer, num_epochs=25):
    since = time()

    loss_change = []

    model.train()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        start = datetime.now()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx % 100 ==0:
                print('[batch {}/{}]'.format(batch_idx, len(dataloader)))

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        finish = datetime.now()
        print(f"epoch {epoch} took ", finish-start)

        epoch_loss = running_loss / len(dataloader.dataset)
        loss_change.append(epoch_loss)

        print('{} Loss: {:.4f}'.format('training:', epoch_loss))
        print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    training_file = open("../statistics/training_time.txt", "w+")
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=training_file)
    training_file.close()

    return model,loss_change

def test(model, device, dataloader):
    since = time()

    accuracy = 0
    predicted = []
    actual = []

    model.eval()

    running_corrects = 0
    wrong_images_limit = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx % 100 ==0:
            print('[batch {}/{}]'.format(batch_idx, len(dataloader)))

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

        if wrong_images_limit<4:
            for p, r, img in zip(preds, labels.data, inputs):
                if p != r:
                    # print(img.cpu().numpy().shape)
                    plt.imsave(f"r{r}p{p}.png",img.cpu().numpy()[0], cmap="gray", format="png")
                    wrong_images_limit+=1
                    if wrong_images_limit>4:
                        break

        batch_pred = np.asarray(preds.to("cpu")).tolist()
        batch_actu = np.asarray(labels.data.to("cpu")).tolist()

        predicted.extend(batch_pred)
        actual.extend(batch_actu)

    accuracy = running_corrects.double() / len(dataloader.dataset)

    print('Accuracy: {:.4f}'.format(accuracy))
    print()

    test_confusion_matrix = confusion_matrix(actual, predicted, labels=[x for x in range(0,10)])

    time_elapsed = time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print()
    return model, accuracy, test_confusion_matrix

def get_params_to_update(model_ft, feature_extract):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    return params_to_update
