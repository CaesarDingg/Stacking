import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch import optim
import torch.nn.functional as F
# from adaboost_tool import *
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from models import *
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def training(epochs, fold_train, net, lr, weight_decay, i, le):
    fold_train_temp = copy.deepcopy(fold_train)
    del fold_train_temp[i]
    
    print(fold_train_temp)
    
    trainset = torch.utils.data.ConcatDataset(fold_train_temp)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x_train, y_train) in enumerate(trainloader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            output = net(x_train)
            loss = criterion(output, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

        print('Model net%d' % i)
        print('Epoch %d | Loss: %.5f, Acc: %5f%% (%d/%d)' % (epoch, (train_loss/(batch_idx+1)), 100*correct/total, correct, total))
    print('saving...')
    torch.save(net, './stacking_net%d_%d.pkl' % le, i)


def get_train_pred(flod_train, model_list):       # get layer1 model predict
    pred_result = torch.zeros((50000, 30)).to(device)
    a = 0
    for le in range(3):
        for i in range(5):
            trainloader = torch.utils.data.DataLoader(flod_train[i], batch_size=100, num_workers=4)
            for idx, (x, y) in enumerate(trainloader):
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    pred = model_list[a](x)
                    if a < 14:
                        a += 1

                pred_result[i*100:i*100+100, le*10:le*10+10] = pred

        # pred_result = pred_result[1:, :]
#     pred_result = pred_result.cpu().numpy()
#     pred_result = np.delete(pred_result, [0])
    return pred_result


# def get_test_pred(testset, model_list):            # get test predict and compute average
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=4)
#     test_result = torch.ones((1, 10)).to(device)
#     for idx, (x, y) in enumerate(testloader):
#         x, y = x.to(device), y.to(device)
#         pred_result = torch.zeros((1, 10)).to(device)
#         for i in range(len(model_list)):
#             model = model_list[i]
#             with torch.no_grad():
#                 pred = model(x)
#             pred_result = (pred_result + pred) / len(model_list)
# #         test_result = pred_result / len(model_list)
#
#         test_result = torch.cat((test_result, pred_result), 0)
#     return test_result[1:, :]


def training_layer2(pred_result, trainset, net, epochs, lr, weight_decay):
    # 自定义数据集训练layer2 model
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, num_workers=4)
    y_label = torch.ones((1,)).cuda()
    for i, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        y_label = torch.cat((y_label, y), 0)

    y_label = y_label[1:]
    y_label = y_label.type(torch.LongTensor)

    trainset2 = Data.TensorDataset(pred_result, y_label)
    trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=32)

    # training...
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(net.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x_train, y_train) in enumerate(trainloader2):
            x_train, y_train = x_train.to(device), y_train.to(device)
            output = net(x_train)
            loss = criterion(output, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

        print('Model net layer2')
        print('Epoch %d | Loss: %.5f, Acc: %5f%% (%d/%d)' % (
        epoch, (train_loss / (batch_idx + 1)), 100 * correct / total, correct, total))
    print('saving...')
    torch.save(net, './stacking_net_layer2.pkl')
