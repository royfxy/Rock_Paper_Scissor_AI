from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt


dataset = 'rock_paper_scissor/network/gesture_recognition/data/keypoint_spc_xyz.csv'

whole_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32')

np.random.shuffle(whole_dataset)

X_dataset = whole_dataset[:, 1:]
y_dataset = whole_dataset[:, :1].astype('int32').reshape(len(whole_dataset),)


class Hand_MLP(nn.Module):
    """MLP encoder module."""

    def __init__(self, n_in, n_hid, n_out):
        super(Hand_MLP, self).__init__()
        self.dropout_1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(n_in, n_hid, bias=True)
        self.dropout_2 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(n_hid, 10, bias=True)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(10, n_out, bias=True)
        self.softmax = nn.Softmax(dim=1)

        # self.dropout_2 = nn.Dropout(p = 0.5)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # print(x.size())
        # x = self.dropout_2(x)
        x = self.dropout_1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.fc2(x)
        # x = self.dropout_2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        # x = F.relu(self.fc1(x))
        # x = self.dropout_2(x)
        # x = self.fc2(x)
        return x


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


acc = 0
total_result = len(X_dataset)
correct_result = 0
correct_result_died = 0

dead_index = []

dataset_name = 'Hand'


def draw_curve(x_epoch, epoch_accuracy, epoch_loss, val_acc, val_loss):
    # x_epoch.append(current_epoch)
    x_size = x_epoch
    x_length = list(range(x_size))
    # print(z)
    # print(len(x_length))
    # figure(figsize=(8, 6), dpi=300)
    fig = plt.figure()
    # fig(figsize=(8, 6), dpi=300)
    ax0 = fig.add_subplot(221, title="Train_accuracy")
    ax1 = fig.add_subplot(222, title="Train_loss")
    ax2 = fig.add_subplot(223, title="Val_acc")
    ax3 = fig.add_subplot(224, title="Val_loss")
    ax0.plot(x_length, epoch_accuracy, 'b-', label='Train_accuracy')
    ax1.plot(x_length, epoch_loss, 'r-', label='Train_loss')
    ax2.plot(x_length, val_acc, 'm-', label='Val_acc')
    ax3.plot(x_length, val_loss, 'g-', label='Val_loss')
    plt.tight_layout()
    plt.show(fig)


def get_validation_loss(model, Val, valid_dataloader):

    model.eval()
    val_loss = []
    accuracy = 0

    # In my case I used CrossEntropy loss and enable it to GPU device
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss().to(device)

    for idx, (data, labels) in enumerate(valid_dataloader):

        data, labels = Variable(data), Variable(labels.long())
        output = model(data)
        loss = criterion(output, labels)
        val_loss.append(loss.item())
        val_acc = ((output.argmax(dim=1) == labels).float().mean())
        accuracy += val_acc/len(Val)

    return np.mean(np.array(val_loss)), accuracy


def train_model_static(recognition_model_pth, epoch_num=200):
    best_model = None
    min_epochs = 5
    min_val_loss = 5

    n = len(X_dataset)

    split_index = n // 5
    train_index = split_index * 3

    train_x = X_dataset[:train_index, :]
    train_y = y_dataset[:train_index]

    valid_x = X_dataset[train_index:, :]
    valid_y = y_dataset[train_index:]

    train_dataset = SimpleDataset(train_x, train_y)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0)
    valid_dataset = SimpleDataset(valid_x, valid_y)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=True)

    model = Hand_MLP(63, 30, 3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    compute_loss = nn.CrossEntropyLoss()
    epoch_a = []
    epoch_l = []
    val_a = []
    val_l = []

    for epoch in tqdm(range(epoch_num), ascii=True):

        epoch_train_loss = []
        epoch_accuracy = 0
        epoch_loss = 0

        for idx, (data, labels) in enumerate(train_dataloader):
            data, labels = Variable(data), Variable(labels.long())

            optimizer.zero_grad()

            logits = model(data)
            labels = torch.flatten(labels)
            loss = compute_loss(logits, labels)

            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())
            # print(output.shape)
            acc = ((logits.argmax(dim=1) == labels).float().mean())
            epoch_accuracy += acc/len(train_dataloader)
            epoch_loss += loss/len(train_dataloader)

        # print('\n Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        val_loss, val_acc = get_validation_loss(
            model, valid_dataloader, valid_dataloader)

        model.train()
        epoch_a.append(epoch_accuracy)
        epoch_l.append(epoch_loss.detach().numpy())
        val_a.append(val_acc)
        val_l.append(val_loss)

        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            # print('pass here')
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

            # draw_curve(epoch+1,epoch_a,epoch_l,val_a,val_l)
            tqdm.write('Best model at Epoch {:03d} train_loss {:.5f} val_loss {:.5f} val_acc {:.5f}'.format(
                epoch, np.mean(np.array(epoch_train_loss)), val_loss, val_acc))

    # Change the name based on you decision
    torch.save(best_model.state_dict(), recognition_model_pth)
