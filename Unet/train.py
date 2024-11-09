import os
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import UNet
from data_loader import *
from torchvision.utils import save_image



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = 'D:/Python/Python project/My_project/Unet/data/VOC/VOCdevkit/VOC2007'
save_path = 'train_image'

if __name__ == '__main__':
    num_classes = 2   # 2个类别+1个背景
    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
    net = UNet(num_classes).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('load weight successful')
    else:
        print('weight not found')
    
    opt = optim.Adam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    epoch = 1
    while epoch < 200:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            
            out_image = net(image)
            train_loss = loss_fn(out_image, segment_image.long())
            
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            
            if i % 5 == 0:
                print(f'epoch:{epoch}, step:{i}, train_loss:{train_loss.item()}')


            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image.long()], dim=0)
            save_image(img, f'{save_path}/{i}.png')

        if epoch % 20 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully')
        epoch += 1      
