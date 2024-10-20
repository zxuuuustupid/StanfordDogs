import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from stanford.CNN import CNN
from stanford.dataset import SimpleImageFolderDataset

root_dir = './Images/train'
root_dir2='./Images/test'
# 定义变换操作
transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    # 创建Dataset实例
    dataset = SimpleImageFolderDataset(root_dir, transform=transform)
    dataset2 = SimpleImageFolderDataset(root_dir2, transform=transform)

    # 创建DataLoader实例
    data_loader1 = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    data_loader2 = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda:0")
    print("load the model...")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9,nesterov=True,weight_decay=5e-4)

    # 现在你可以在训练循环中使用data_loader了
    for epoch in range(101):
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

        print("Epoch:", epoch, "\nloss:", train_loss, "train acc:", train_acc / len(data_loader1),"test acc:",test_acc/len(data_loader2), "\n")
