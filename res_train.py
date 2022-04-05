import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from project1_model import project1_model
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') 

## THIS DATA AUMENTATION TEND TO CAUSE OVERFITTNG
# Random Erasing, cite from https://github.com/zhunzhong07/Random-Erasing/blob/ed05424dcb2fd502eafb41b2b1151fb7cd16cbcd/cifar.py#L26
parser.add_argument('--p', default=0, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')
args = parser.parse_args()

# Hyperparameter
EPOCH =  150
pre_epoch = 0  
BATCH_SIZE = 128    
 
LR = 0.01    


# Data Prepocessing and Augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    transforms.RandomErasing(p = 0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3) ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


PATH = 'project1_model.pt'

# project1_model
net = project1_model().to(device)

# # load the pre-trained model
net.load_state_dict(torch.load(PATH))
net.eval()


criterion = nn.CrossEntropyLoss()  
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) 
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9,0.999),eps=1e-08, weight_decay=0)
#optimizer = optim.RMSprop(net.parameters(), lr=LR, alpha=0.99, eps=1e-08, weight_decay=5e-4,momentum=0)

# lr list
lr_list=[]

    


# Training
if __name__ == "__main__":
    if not os.path.exists(args.outf):

        os.makedirs(args.outf)
    best_acc = 0  #initialize best test accuracy

    train_epoch = []
    train_acc = []
    train_losses = []

    test_epoch = []
    test_acc = []
    test_losses = []
    

    print("Start Training, Resnet!") 
    with open("acc.txt", "w", encoding='utf-8') as f:
        with open("log.txt", "w", encoding='utf-8')as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                
                # Manually adjust the learning rate
                if epoch % 5==0:
                    for p in optimizer.param_groups:
                        p['lr'] *= 0.8
                lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

                for i, data in enumerate(trainloader, 0):
                    
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()


                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()

                    #train_epoch.append(epoch+1)
                    #train_acc.append((100. * correct / total))
                    #train_losses.append((sum_loss / (i + 1)))

                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                        % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                        % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # Convert tensor to numpy  
                correct = correct.cpu().numpy()
                
                train_epoch.append(epoch+1)
                train_acc.append((100. * correct / total))
                train_losses.append((sum_loss / (i + 1)))

                # Test accuracy each epoch
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for i, data in enumerate(testloader, 0):
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)

                        # Compute test loss
                        loss = criterion(outputs, labels)
                        sum_loss += loss.item()

                        # Get the predicted class index 
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()

                        #test_epoch.append(epoch+1)
                        #test_acc.append((100. * correct / total))
                        #test_losses.append((sum_loss / (i + 1)))

                    print('test accuracy: %.3f%%' % (100. * correct / total))
                    acc = 100. * correct / total
                    # put test accuracy (per iteration) into acc.txt
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

                    # convert tensor to numpy  
                    correct = correct.cpu().numpy()

                    test_epoch.append(epoch+1)
                    test_acc.append((100. * correct / total))
                    test_losses.append((sum_loss / (i + 1)))


                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w", encoding='utf=8')
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc

                        # save highest test accuracy model
                        # torch.save(net.state_dict(), "project1_model.pt")



            print("Training Finished, TotalEPOCH=%d" % EPOCH)
            print("train_epoch size:", len(train_epoch),"    ","test_epoch size:",len(test_epoch))

            # plot losses and accuracy
            # Accuracy
            plt.figure(1)
            plt.subplot(1, 3, 1)
            plt.plot(train_epoch, train_acc,'g', label='Train accuracy')
            plt.plot(test_epoch, test_acc,'b', label='Test accuracy')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy %')
            plt.title("Accuracy")

            # Losses
            plt.figure(1)
            plt.subplot(1, 3, 2)
            plt.plot(train_epoch,train_losses,'g', label='Train Losses')
            plt.plot(test_epoch,test_losses,'b', label='Test Losses')
            plt.legend()
            plt.title("Losses")
            plt.xlabel('Epochs')
            plt.ylabel('Losses %')

            # Learning rate
            plt.figure(1)
            plt.subplot(1, 3, 3)
            plt.plot(train_epoch,lr_list,'g', label='Learning Rate')
            plt.legend(loc='upper right')
            plt.title("Learning Rate")
            plt.xlabel('Epochs')
            plt.ylabel('LR')

            plt.show()