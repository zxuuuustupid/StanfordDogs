import os
import shutil
import random


def split_dataset(base_dir, train_dir, test_dir, split_ratio=0.8):
    """
    将图片数据集按比例划分为训练集和测试集。

    :param base_dir: 数据集根目录（包含多个类别文件夹）
    :param train_dir: 训练集输出目录
    :param test_dir: 测试集输出目录
    :param split_ratio: 训练集占比（默认4:1比例，split_ratio=0.8）
    """
    # 确保输出目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历每个类别文件夹
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)

        if not os.path.isdir(category_path):
            continue

        # 创建类别对应的训练集和测试集文件夹
        train_category_path = os.path.join(train_dir, category)
        test_category_path = os.path.join(test_dir, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)

        # 获取该类别下的所有图片文件
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        random.shuffle(images)  # 随机打乱

        # 按比例划分
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # 复制图片到对应目录
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_category_path, img))
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_category_path, img))

        print(f"分类 {category}: 训练集 {len(train_images)} 张, 测试集 {len(test_images)} 张")


if __name__ == "__main__":
    # 定义路径
    base_dir = "images/images"  # 原始数据集目录
    train_dir = "train"  # 训练集输出目录
    test_dir = "test"  # 测试集输出目录

    # 执行划分
    split_dataset(base_dir, train_dir, test_dir)
