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

root = "/home/zlab/zhangshun/torch1/data_et/"
path = "/home/zlab/zhangshun/torch1/modelparahamm256.pth"
model = Net()

model.load_state_dict(torch.load(path))
for para in model.parameters():
    para.requires_grad = False

model.cuda()



evaluate_data = MyDataset(txt=root + 'evaluate.txt', transform=transforms.ToTensor())
evaluate_loader = DataLoader(dataset=evaluate_data, batch_size=32)


f4 = open(root + 'evaluate_acc.txt', 'w')
print(model)
eval_acc = 0.
# evaluation--------------------------------
model.eval()
for batch_x, batch_y in evaluate_loader:
    batch_x, batch_y = Variable(batch_x.cuda(), volatile=True), Variable(batch_y.cuda(), volatile=True)
    out = model(batch_x)
    pred = torch.max(out, 1)[1]
    print(pred)
    num_correct = (pred == batch_y).sum()
    eval_acc += num_correct.data.item()
    print(eval_acc)
    f4.write(str(eval_acc / (len(evaluate_data))) + '\n')
f4.close()
