import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split

from sklearn.model_selection import KFold

from rock_paper_scissor.network.gesture_prediction.dataset import RockPaperScissorDataset, read_data

from torchvision import transforms

from rock_paper_scissor.network.gesture_prediction.model import Model

from torch.utils.tensorboard import SummaryWriter


criterion = nn.NLLLoss()    

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total_count = 0
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_count += len(data)
        losses.append(loss.item())
    loss = sum(losses)/len(losses)
    return correct/total_count, loss
    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_count += len(data)

    test_loss /= total_count
    accuracy = 100. * correct / total_count

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total_count,
        accuracy))
    
    return accuracy, test_loss

def train_model(file_pth, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        # convert to tensor
        transforms.ToTensor(),
    ])

    data, labels = read_data()

    dataset = RockPaperScissorDataset(data, labels, transform=transform)

    # split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

        accuracy, loss = test(model, device, test_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), file_pth)

def k_fold_cross_validation_train(folds, file_pth, epochs=100):
    writer = SummaryWriter()
    data, labels = read_data()
    transform = transforms.Compose([
        # convert to tensor
        transforms.ToTensor(),
    ])
    dataset = RockPaperScissorDataset(data, labels, transform=transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_folds = KFold(n_splits=folds, shuffle=True)
    
    models = []
    optimizers = []
    train_loaders = []
    test_loaders = []
    for fold, (train_ids, test_ids) in enumerate(k_folds.split(dataset)):
        models.append(Model().to(device))
        optimizers.append(optim.Adam(models[-1].parameters(), lr=0.001))
        train_sampler = SubsetRandomSampler(train_ids)
        test_sampler = SubsetRandomSampler(test_ids)
        train_loaders.append(DataLoader(dataset, batch_size=16, sampler=train_sampler))
        test_loaders.append(DataLoader(dataset, batch_size=16, sampler=test_sampler))
    best_accuracy = 0
    best_accuracy_index = 0
    for epoch in range(1, epochs + 1):
        test_accuracies = []
        test_losses = []
        train_accuracies = []
        train_losses = []
        for fold, _ in enumerate(k_folds.split(dataset)):
            train_accuracy, train_loss = train(models[fold], device, train_loaders[fold], optimizers[fold], epoch)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            test_accuracy, test_loss = test(models[fold], device, test_loaders[fold])
            test_accuracies.append(test_accuracy)
            test_losses.append(test_loss)
        print(f"Epoch {epoch} accuracy: {sum(test_accuracies)/len(test_accuracies)}")
        writer.add_scalar("Accuracy/train", sum(train_accuracies)/len(train_accuracies), epoch)
        writer.add_scalar("Accuracy/test", sum(test_accuracies)/len(test_accuracies), epoch)
        writer.add_scalar("Loss/train", sum(train_losses)/len(train_losses), epoch)
        writer.add_scalar("Loss/test", sum(test_losses)/len(test_losses), epoch)
        if sum(test_accuracies)/len(test_accuracies) > best_accuracy:
            best_accuracy = sum(test_accuracies)/len(test_accuracies)
            best_accuracy_index = epoch
            torch.save(models[0].state_dict(), file_pth)
            print("best accuracy achieved: ", best_accuracy)
    print(f"Best accuracy: {best_accuracy}")