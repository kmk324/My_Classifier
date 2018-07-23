import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='./my_traindata', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./my_testdata', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

classes = ('elephant','tiger') #my dataset_classes name. you can edit.


#Convolution Neural Network Definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#LossFunction and Optimizer definition
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#model train
for epoch in range(100):  # Iterative learning, you can edit this number '100' to what you want.

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # after inputs
        inputs, labels = data

        # Initialize gradient to '0'
        optimizer.zero_grad()

        # forward + backward + optimizer
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistic output
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20mini-batches , you can edit this number '20' to what you want.
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20)) # you can edit this number '20' to what you want.
            running_loss = 0.0
print('Finished Training')



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 20 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(2)) #this number '2'  is your dataset's  number of classes.
class_total = list(0. for i in range(2)) #this number '2'  is your dataset's  number of classes.
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(1): #this number '2'  is (your dataset's  number of classes) - 1.
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2): #this number '2'  is your dataset's  number of classes.
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
