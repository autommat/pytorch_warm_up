from multiprocessing.dummy import freeze_support

import pandas
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from experiments.common import initialize_model, train, test, OwnNet, get_train_loader_own, get_test_loader_own, \
    get_params_to_update, get_test_loader_squeeze, get_train_loader_squeeze

if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10
    batch_size = 32
    test_batch_size = 64
    num_epochs = 3 # todo: change back to 15

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True


    # Initialize the model for this run
    model_ft = initialize_model(num_classes, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)
    print(model_ft)


    print("Initializing Datasets and Dataloaders...")


    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = get_params_to_update(model_ft, feature_extract)


    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    train_dataloader = get_train_loader_squeeze(batch_size)
    test_dataloader = get_test_loader_squeeze(batch_size)
    model_ft, hist_loss = train(model_ft, device, train_dataloader, criterion, optimizer_ft, num_epochs=num_epochs)
    model_ft, acc, cm = test(model_ft, device, test_dataloader, criterion, optimizer_ft)



    # Initialize the non-pretrained version of the model used for this run
    # full_model = initialize_model(num_classes, feature_extract=False, use_pretrained=False)
    # full_model = full_model.to(device)
    # full_optimizer = optim.SGD(full_model.parameters(), lr=0.001, momentum=0.9)
    # full_criterion = nn.CrossEntropyLoss()
    # full_model,hist_loss = train(full_model, device, train_dataloader, full_criterion, full_optimizer, num_epochs=num_epochs)
    # full_model, acc, cm = test(full_model, device, train_dataloader, full_criterion, full_optimizer)

    own_net_batch_size = 64
    own_net_test_batch_size = 100
    own_net_model = OwnNet().to(device)
    train_loader = get_train_loader_own(own_net_batch_size)
    test_loader = get_test_loader_own(own_net_test_batch_size)
    # own_net_optimizer = optim.SGD(own_net_model.parameters(), lr=0.0001, momentum=0.9) #ex1
    own_net_optimizer = optim.SGD(own_net_model.parameters(), lr=0.01, momentum=0.9) #ex2
    own_criterion = nn.CrossEntropyLoss()
    # own_net_model, hist_loss = train(own_net_model, device, train_loader, own_criterion, own_net_optimizer, num_epochs=num_epochs)
    model_ft, shist, cm = test(own_net_model, device, test_loader, own_criterion, own_net_optimizer)


    cm_file = open('cm_own1.txt', 'w+')
    cm_pd = pandas.DataFrame(data=cm)
    cm_pd = cm_pd.rename_axis("true", axis="columns")
    cm_pd = cm_pd.rename_axis("pred", axis="rows")
    print(cm_pd, file=cm_file)
    cm_file.close()

    plt.xlabel("Training Epochs")
    plt.ylabel("Loss function value")
    plt.plot(range(1,num_epochs+1),hist_loss)
    plt.ylim((0,max(hist_loss)+0.1))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.savefig(f'loss{datetime.now().hour}-{datetime.now().minute}.png', format="png")
