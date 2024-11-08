import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models
from models.CNN import CNN
from models.VGG import VGG
from models.ResNet import ResNet18, ResNet50
from dataset import SimpleImageFolderDataset
from draw import process_show

root_dir = './Images/train'
root_dir2 = './Images/test'
model_name = 'Pretrained resnet'
epoch_num = 20

# 定义变换操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),  # 随机水平镜像
    transforms.RandomRotation(10),
])

if __name__ == '__main__':
    # 创建Dataset实例
    dataset = SimpleImageFolderDataset(root_dir, transform=transform)
    dataset2 = SimpleImageFolderDataset(root_dir2, transform=transform)

    # 创建DataLoader实例
    data_loader1 = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    data_loader2 = DataLoader(dataset2, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda:0")
    print("load the model...")

    weights = ResNet18_Weights.DEFAULT  # 使用默认的预训练权重
    model = models.resnet18(weights=weights).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 先冻结卷积层，只训练全连接层
    for param in model.parameters():
        param.requires_grad = False

    # 替换全连接层
    fc_in_features = model.fc.in_features
    model.fc = nn.Linear(fc_in_features, 10, bias=True).to(device)  # 10类（斯坦福狗数据集）

    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4)

    loss_list = []
    train_acclist = []
    test_acclist = []

    # 第一阶段：冻结卷积层，训练全连接层
    for epoch in range(1, epoch_num + 1):
        train_loss = 0
        train_acc = 0
        model.train()

        for images, label in data_loader1:
            img = images.to(device)
            label = label.to(device)

            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc

        test_acc = 0
        model.eval()
        for images2, label2 in data_loader2:
            img = images2.to(device)
            label = label2.to(device)

            out = model(img)
            loss = criterion(out, label)

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            test_acc += acc

        train_acclist.append(train_acc / len(data_loader1))
        test_acclist.append(test_acc / len(data_loader2))
        loss_list.append(train_loss)

        print("Epoch:", epoch, "loss:{:.4f}  train acc:{:.4f}  test acc:{:.4f}".format(
            train_loss, train_acc / len(data_loader1), test_acc / len(data_loader2)))

    # 第二阶段：解冻深层卷积层（layer4），继续训练
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 重新设置优化器，包含解冻的层
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9, nesterov=True, weight_decay=5e-4)

    # 继续训练
    for epoch in range(epoch_num + 1, epoch_num + 21):  # 继续训练20个epoch
        train_loss = 0
        train_acc = 0
        model.train()

        for images, label in data_loader1:
            img = images.to(device)
            label = label.to(device)

            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc

        test_acc = 0
        model.eval()
        for images2, label2 in data_loader2:
            img = images2.to(device)
            label = label2.to(device)

            out = model(img)
            loss = criterion(out, label)

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            test_acc += acc

        train_acclist.append(train_acc / len(data_loader1))
        test_acclist.append(test_acc / len(data_loader2))
        loss_list.append(train_loss)

        print("Epoch:", epoch, "loss:{:.4f}  train acc:{:.4f}  test acc:{:.4f}".format(
            train_loss, train_acc / len(data_loader1), test_acc / len(data_loader2)))

    # 绘制训练过程中的损失和准确率曲线
    process_show(model_name, list(range(1, epoch_num + 1)), loss_list, train_acclist, test_acclist)
