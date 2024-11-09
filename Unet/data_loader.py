import os
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *
import numpy as np
import torch
from torch.nn.functional import one_hot

transform = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))


    def __len__(self):
        return len(self.name)

    
    def __getitem__(self, index):
        segment_name = self.name[index]  # 格式：'2007_000032.png'
        # 转换为‘xxx.jpg’
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name[:-4] + '.jpg')
        # 上一句也可以写成：image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('.png', '.jpg'))

        # 读取图片
        # 统一图片大小：做一个mask,将原图贴到mask上，然后resize
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open_rgb(image_path)

        return transform(image), torch.Tensor(np.array(segment_image))


if __name__ == '__main__':
    data = MyDataset("D:/Python/Python project/My_project/Unet/data/VOC/VOCdevkit/VOC2007")
    print(data[0][0].shape)
    print(data[0][1].shape)
    out = one_hot(data[0][1].long())
    print(out.shape)