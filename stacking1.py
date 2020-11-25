import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision
from model import *
from stacking_tool import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train1, train2, train3, train4, train5 = torch.utils.data.random_split(
    dataset=trainset, lengths=[10000, 10000, 10000, 10000, 10000])

flod_train = [train1, train2, train3, train4, train5]


net = VGG('VGG16').cuda()

for le in range(3):         # layer1 base model training
    for i in range(5):
        training(40, flod_train, net, 0.001, 5e-4,  i, le)

net2 = Net2(30, 10, 10).cuda()   # layer2 model

learners = {}
a = 0
for le in range(3):
    for i in range(5):
        learners[a] = torch.load('stacking_net%d_%d.pkl' % (le, i)).eval().cuda()
        a += 1

pred_result = get_train_pred(flod_train, learners)
# test_result = get_test_pred(testset, learners)
training_layer2(pred_result, trainset, net2, epochs=200, lr=0.0001, weight_decay=1e-6)
