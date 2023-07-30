import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import MSCMRsegDatasetTrain, MSCMRsegDatasetPredict
from model import my_net
import os


def eval(model, dataloader):
    total = 0
    correct = 0
    for data in dataloader:
        inputs = data[0].to(device).float()
        labels = data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


NAME = 'LGE'
BATCH_SIZE = 32

net = my_net()
net.load_state_dict(torch.load('./data/checkpoints/C0/model_latest.pt'))

conv_layer = [0, 4, 8]

dataset_train = MSCMRsegDatasetTrain(NAME, 'train')

dataset_val = MSCMRsegDatasetTrain(NAME, 'val')

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)



device = 'cuda'
net = net.to(device)
criterion = nn.CrossEntropyLoss()
num_batches = len(dataloader_train)
best_acc = 0

for i in conv_layer:
    for param in net[i].parameters():
        param.requires_grad=False

optimizer = optim.SGD(net.parameters(), lr=0.1)

loss_list = []
train_acc = []
val_acc = []

for epoch in range(5):
    total_loss = 0.0
    correct = 0
    total = 0
    for data in dataloader_train:
        inputs = data[0].to(device).float()
        labels = data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = eval(net, dataloader_val)

    if acc > best_acc:
        torch.save(net.state_dict(), f'./data/checkpoints/{NAME}/model_best.pt')
        best_acc = acc
    torch.save(net.state_dict(), f'./data/checkpoints/{NAME}/model_latest.pt')

    loss_list.append(total_loss / num_batches)
    train_acc.append(correct / total)
    val_acc.append(acc)

    print(f'Epoch{epoch}: {total_loss / num_batches}, {correct / total}, {acc}')



for i in conv_layer:
    for param in net[i].parameters():
        param.requires_grad=True

optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(5,15):
    total_loss = 0.0
    correct = 0
    total = 0
    for data in dataloader_train:
        inputs = data[0].to(device).float()
        labels = data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = eval(net, dataloader_val)

    if acc > best_acc:
        torch.save(net.state_dict(), f'./data/checkpoints/{NAME}/model_best.pt')
        best_acc = acc
    torch.save(net.state_dict(), f'./data/checkpoints/{NAME}/model_latest.pt')

    loss_list.append(total_loss / num_batches)
    train_acc.append(correct / total)
    val_acc.append(acc)

    print(f'Epoch{epoch}: {total_loss / num_batches}, {correct / total}, {acc}')


torch.save(loss_list, f'train_loss_{NAME}.pt')
torch.save(train_acc, f'train_acc_{NAME}.pt')
torch.save(val_acc, f'val_acc_{NAME}.pt')
