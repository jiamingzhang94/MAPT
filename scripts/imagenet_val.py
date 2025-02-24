import os
import shutil

# 源数据集路径和目标路径
source_dir = '/data/scratch/datasets/ImageNet/ILSVRC/Data/CLS-LOC/val'
target_dir = '/data/gpfs/projects/punim0619/jiaming/datasets/imagenet/images/val'

# 确保目标目录存在，如果不存在则创建
os.makedirs(target_dir, exist_ok=True)

# 遍历源目录中的每个子文件夹（类别文件夹）
for category_folder in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category_folder)

    # 确保是文件夹
    if os.path.isdir(category_path):
        # 获取文件夹中的所有图像文件（假设都是图像文件）
        image_files = os.listdir(category_path)

        # 选择第一张图像
        if image_files:
            first_image = image_files[0]
            first_image_path = os.path.join(category_path, first_image)

            # 目标路径
            target_category_path = os.path.join(target_dir, category_folder)

            # 确保目标类别文件夹存在
            os.makedirs(target_category_path, exist_ok=True)

            # 复制第一张图像到目标目录
            shutil.copy(first_image_path, target_category_path)
            print(f"Copied {first_image} to {target_category_path}")
