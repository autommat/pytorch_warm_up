from multiprocessing.dummy import freeze_support

import pandas
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
import numpy as np
from experiments.common import initialize_model, train, test, OwnNet, get_train_loader_own, get_test_loader_own, \
    get_params_to_update, get_test_loader_squeeze, get_train_loader_squeeze

if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_num = input("podaj numer eksperymentu:")
    if exp_num == "1" or exp_num == "2":
        batch_size = 64
        test_batch_size = 64
        num_epochs = 20
        print("Inicjalizacja modelu")
        model = OwnNet().to(device)
        if exp_num == "1":
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        print("Inicjalizacja dataloadera")
        train_loader = get_train_loader_own(batch_size)
        test_loader = get_test_loader_own(test_batch_size)
        criterion = nn.CrossEntropyLoss()
    elif exp_num == "3":    #nauka całej sieci
        batch_size = 80
        test_batch_size = 80
        num_epochs = 12
        num_classes = 10
        print("Inicjalizacja modelu")
        model = initialize_model(num_classes, feature_extract=False, use_pretrained=False)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        print("Inicjalizacja dataloadera")
        train_loader = get_train_loader_squeeze(batch_size)
        test_loader = get_test_loader_squeeze(batch_size)
    elif exp_num == "4":    #nauka części sieci
        batch_size = 80
        test_batch_size = 80
        num_epochs = 2 #todo
        num_classes = 10
        print("Inicjalizacja modelu")
        model = initialize_model(num_classes, feature_extract=True, use_pretrained=True)
        model = model.to(device)
        params_to_update = get_params_to_update(model, True)
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        print("Inicjalizacja dataloadera")
        train_loader = get_train_loader_squeeze(batch_size)
        test_loader = get_test_loader_squeeze(batch_size)
    else:
        print("należy podać liczbę między 1 a 4")
        exit(1)

    print(model)

    model, hist_loss = train(model, device, train_loader, criterion, optimizer, num_epochs=num_epochs)
    model, acc, cm = test(model, device, test_loader)


    cm_file = open('../statistics/cm.txt', 'w+')
    cm_pd = pandas.DataFrame(data=cm)
    cm_pd = cm_pd.rename_axis("true", axis="columns")
    cm_pd = cm_pd.rename_axis("pred", axis="rows")
    print(cm_pd, file=cm_file)
    cm_file.close()

    acc_file = open('../statistics/acc.txt', 'w+')
    print(acc, file=acc_file)
    acc_file.close()

    plt.xlabel("Training Epochs")
    plt.ylabel("Loss function value")
    plt.plot(range(1,num_epochs+1),hist_loss)
    plt.ylim((0,max(hist_loss)+0.1))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.savefig(f'loss_change.png', format="png")

    model, acc, cm = test(model, device, train_loader)
    cm_filetr = open('../statistics/cmtr.txt', 'w+')
    cm_pd = pandas.DataFrame(data=cm)
    cm_pd = cm_pd.rename_axis("true", axis="columns")
    cm_pd = cm_pd.rename_axis("pred", axis="rows")
    print(cm_pd, file=cm_filetr)
    cm_filetr.close()

    acc_filetr = open('../statistics/acctr.txt', 'w+')
    print(acc, file=acc_filetr)
    acc_filetr.close()
