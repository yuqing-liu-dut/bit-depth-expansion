from cv2 import imwrite
import torch
import torch.nn as nn


class RK4(nn.Module):
    def __init__(self):
        super(RK4, self).__init__()
        self.f1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=3//2),
                                  nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=3//2),)
        self.f2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=3//2),
                                  nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=3//2),)
        self.f3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=3//2),
                                  nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=3//2),)
        self.f4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=3//2),
                                  nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=3//2),)
    def forward(self, x):
        k1 = self.f1(x)
        k2 = self.f2(x + k1/2)
        k3 = self.f3(x + k2/2)
        k4 = self.f4(x + k3)
        return x + (k1 + 2*k2 + 2*k3 + k4) / 6


class RK4_Explicit(nn.Module):
    def __init__(self):
        super(RK4_Explicit, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=3//2),
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 1),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=3//2),
        )
        self.f3 = nn.Sequential(
            nn.Conv2d(64 * 3, 64, 1),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=3//2),
        )
        self.f4 = nn.Sequential(
            nn.Conv2d(64 * 4, 64, 1),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=3//2),
        )
        self.comb = nn.Conv2d(64 * 4, 64, 1)
    def forward(self, x):
        k1 = self.f1(x)
        k2 = self.f2(torch.cat((x, k1), 1))
        k3 = self.f3(torch.cat((x, k1, k2), 1))
        k4 = self.f4(torch.cat((x, k1, k2, k3), 1))
        return x + self.comb(torch.cat((k1, k2, k3, k4), 1))


class Prox(nn.Module):
    def __init__(self):
        super(Prox, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=3//2),
        )
    def forward(self, x):
        return x + self.conv(x)


class Step(nn.Module):
    def __init__(self):
        super(Step, self).__init__()
        self.RK = RK4()
        self.Proximal = Prox()
    def forward(self, x):
        x = self.RK(x)
        x = self.Proximal(x)
        return x


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(64, 64, 3, padding=3//2),
                                  nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=3//2),)
    def forward(self, x):
        return self.conv(x)+x


class Net(nn.Module):
    def __init__(self, args=None):
        super(Net, self).__init__()
        self.head = nn.Conv2d(6, 64, 3, padding=3//2)
        self.L1 = nn.Sequential(Step(),
                                nn.Conv2d(64, 64, 3, padding=3 // 2),
                                nn.ReLU(True),
                                nn.Conv2d(64, 64, 3, padding=3 // 2), )
        self.L2 = nn.Sequential(Step(),
                                nn.Conv2d(64, 64, 3, padding=3 // 2),
                                nn.ReLU(True),
                                nn.Conv2d(64, 64, 3, padding=3 // 2), )
        self.L4 = nn.Sequential(Step(), Step(), Step(), Step(), Step(), Step(),
                                nn.Conv2d(64, 64, 3, padding=3 // 2),
                                nn.ReLU(True),
                                nn.Conv2d(64, 64, 3, padding=3 // 2), )
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.up1 = nn.Sequential(nn.Conv2d(64, 64 * 4, 3, padding=3 // 2), nn.PixelShuffle(2))
        self.up2 = nn.Sequential(nn.Conv2d(64, 64 * 4, 3, padding=3 // 2), nn.PixelShuffle(2))
        self.R4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=3 // 2), nn.ReLU(True), nn.Conv2d(64, 64, 3, padding=3 // 2))
        self.R2 = nn.Sequential(nn.Conv2d(128, 64, 1),
                                nn.Conv2d(64, 64, 3, padding=3 // 2), nn.ReLU(True), nn.Conv2d(64, 64, 3, padding=3 // 2))
        self.R1 = nn.Sequential(nn.Conv2d(128, 64, 1),
                                nn.Conv2d(64, 64, 3, padding=3 // 2), nn.ReLU(True), nn.Conv2d(64, 64, 3, padding=3 // 2))

        self.tail_texture = nn.Sequential(nn.Conv2d(64, 64, 3, padding=3//2),
                                          nn.ReLU(True),
                                          nn.Conv2d(64, 3, 3, padding=3//2),)
        self.tail_color = nn.Sequential(nn.Conv2d(64, 64, 3, padding=3//2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 3, 3, padding=3//2),
                                        nn.Sigmoid(),)

    def forward(self, x, octave):
        head = self.head(x)
        L1 = self.L1(head) + head

        L2 = self.down1(L1)
        L2 = self.L2(L2) + L2

        L4 = self.down2(L2)
        L4 = self.L4(L4) + L4

        R4 = self.R4(L4) + L4

        R2 = self.up2(R4)
        R2 = self.R2(torch.cat((L2, R2), 1)) + R2

        R1 = self.up1(R2)
        R1 = self.R1(torch.cat((L1, R1), 1)) + R1

        # texture = self.tail_texture(R1)
        color = self.tail_color(R1)
        # return texture + color * octave
        return x[:, 0:3, :, :] + color * octave

if __name__ == '__main__':
    import os, cv2
    import numpy as np
    from torch.autograd import Variable
    import torch
    from tqdm import tqdm
    
    with torch.no_grad():
        net = Net().cuda()
        net.load_state_dict(torch.load("model_latest.pt"))

        for missing_bits in [1, 2, 3, 4, 5, 6, 7]:

            octave = 2 ** missing_bits

            test_folder = "Kodim\\%d"%missing_bits
            gt_folder = "Kodim\\GT"

            output_folder = "Result\\%d"%missing_bits
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for index, image in tqdm(enumerate(os.listdir(test_folder))):
                
                img = cv2.imread(os.path.join(gt_folder, image))
                cv2.imwrite(os.path.join(output_folder, '%d-HBD.png'%index), img)
                img = cv2.imread(os.path.join(test_folder, image))
                cv2.imwrite(os.path.join(output_folder, '%d-LBD.png'%index), img)

                img = np.transpose(img, (2, 0, 1))
                img = img[np.newaxis, :, :, :]
                img = Variable(torch.FloatTensor(img)).cuda()
                octave_tensor = torch.zeros(img.shape).cuda() + octave
                img_tensor = torch.cat((img, octave_tensor), 1)
                img = net(img_tensor, octave).detach().cpu().numpy()[0]
                img = np.transpose(img, (1, 2, 0))
                cv2.imwrite(os.path.join(output_folder, '%d-RBD.png'%index), img)
