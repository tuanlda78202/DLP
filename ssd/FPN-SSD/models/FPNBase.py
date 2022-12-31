from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision
import torch
from torchvision.models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FPNBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(FPNBase, self).__init__()
        
#         self.resnet = resnet50(pretrained = False, progress = True)
        self.resnet = resnet50(pretrained = True, progress = True)
        # Standard convolutional layers in VGG16
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=3, stride = 1, padding=0).to(device)  # stride = 1, by default

        # lateral connection
        self.lat_p4 = nn.Conv2d(
            in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lat_p3 = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
       

        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, affine=True)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(256, eps=1e-05, affine=True)
        self.conv8 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(256, eps=1e-05, affine=True)
        self.conv9 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        
        return F.upsample(x, size=(H,W), mode='bilinear', align_corners=False) + y
    
    
    def forward(self, x):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        """
        # bottom-up pathway
        h = F.relu(self.resnet.bn1(self.resnet.conv1(x))) # C = 64
        h = F.max_pool2d(h, kernel_size=3, stride=2, padding=1) 
        
        c2 = self.resnet.layer1(h)  #256
        c3 = self.resnet.layer2(c2) #512
        c4 = self.resnet.layer3(c3) #1024
        c5 = self.resnet.layer4(c4) #2048
        
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(self.bn6(p6)))
        p8 = self.conv8(F.relu(self.bn7(p7)))
        p9 = self.conv9(F.relu(self.bn8(p8)))
        
        # top
        p5 = self.toplayer(c5) #256
            
        p4 = self._upsample_add(p5, self.lat_p4(c4)) #256
        p3 = self._upsample_add(p4, self.lat_p3(c3)) #256
        
        p4 = self.smooth1(p4) #256
        p3 = self.smooth2(p3) #256 

        return p3, p4, p5, p6, p7, p8, p9