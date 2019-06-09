import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
root="/home/zlab/zhangshun/torch1/data_et/"

# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        #print(" haha ")
        return img, label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor());
#print(dir(train_data))
test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
#print(dir(train_loader))
test_loader = DataLoader(dataset=test_data, batch_size=32)

#-----------------create the Net and training------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__() #input channal/output channal/kenerl size/stride/padding
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(56320, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


f1 = open(root + 'trainloss.txt', 'w')
f2 = open(root + 'trainacc.txt', 'w')
f3 = open(root + 'testloss.txt', 'w')
f4 = open(root + 'testacc.txt', 'w')


for cycle in range(1):
    model = Net()
    model = model.cuda()
    # print(model)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    # print('cycle {}'.format(cycle + 1))
    for epoch in range(100):
        # print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.data.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f1.write(str(64*train_loss / (len(train_data)))+'\n')
        f2.write(str(train_acc / (len(train_data))) + '\n')
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                train_data)), train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = Variable(batch_x.cuda(), volatile=True), Variable(batch_y.cuda(), volatile=True)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.data.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data.item()
        f3.write(str(64*eval_loss / (len(test_data)))+'\n')
        f4.write(str(eval_acc / (len(test_data))) + '\n')
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_data)), eval_acc / (len(test_data))))
torch.save(model.state_dict(), '/home/zlab/zhangshun/torch1/modelparahann256.pth')
f1.close()
f2.close()
f3.close()
f4.close()
