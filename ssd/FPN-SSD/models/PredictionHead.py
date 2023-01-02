from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'p3': 4,
                   'p4': 6,
                   'p5': 6,
                   'p6': 6,
                   'p7': 4,
                   'p8': 4,
                   'p9': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_p3 = nn.Conv2d(256, n_boxes['p3'] * 4, kernel_size=3, padding=1)
        self.loc_p4 = nn.Conv2d(256, n_boxes['p4'] * 4, kernel_size=3, padding=1)
        self.loc_p5 = nn.Conv2d(256, n_boxes['p5'] * 4, kernel_size=3, padding=1)
        self.loc_p6 = nn.Conv2d(256, n_boxes['p6'] * 4, kernel_size=3, padding=1)
        self.loc_p7 = nn.Conv2d(256, n_boxes['p7'] * 4, kernel_size=3, padding=1)
        self.loc_p8 = nn.Conv2d(256, n_boxes['p8'] * 4, kernel_size=3, padding=1)
        self.loc_p9 = nn.Conv2d(256, n_boxes['p9'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_p3 = nn.Conv2d(256, n_boxes['p3'] * n_classes, kernel_size=3, padding=1)
        self.cl_p4 = nn.Conv2d(256, n_boxes['p4'] * n_classes, kernel_size=3, padding=1)
        self.cl_p5 = nn.Conv2d(256, n_boxes['p5'] * n_classes, kernel_size=3, padding=1)
        self.cl_p6 = nn.Conv2d(256, n_boxes['p6'] * n_classes, kernel_size=3, padding=1)
        self.cl_p7 = nn.Conv2d(256, n_boxes['p7'] * n_classes, kernel_size=3, padding=1)
        self.cl_p8 = nn.Conv2d(256, n_boxes['p8'] * n_classes, kernel_size=3, padding=1)
        self.cl_p9 = nn.Conv2d(256, n_boxes['p9'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, p3_feats, p4_feats, p5_feats, p6_feats, p7_feats, p8_feats, p9_feats):
        """
        Forward propagation.
        :param p3_feats: p3 feature map, a tensor of dimensions (N, 256, 38, 38)
        :param p4_feats: p4 feature map, a tensor of dimensions (N, 256, 19, 19)
        :param p5_feats: p5 feature map, a tensor of dimensions (N, 256, 8, 8)
        :param p6_feats: p6 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param p7_feats: p7 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param p8_feats: p5 feature map, a tensor of dimensions (N, 256, 2, 2)
        :param p9_feats: p8 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = p3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_p3 = self.loc_p3(p3_feats)  # (N, 16, 38, 38)
        l_p3 = l_p3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_p3 = l_p3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_p4 = self.loc_p4(p4_feats)  # (N, 24, 19, 19)
        l_p4 = l_p4.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_p4 = l_p4.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_p5 = self.loc_p5(p5_feats)  # (N, 24, 8, 8)
        l_p5 = l_p5.permute(0, 2, 3, 1).contiguous()  # (N, 8, 8, 24)
        l_p5 = l_p5.view(batch_size, -1, 4)  # (N, 384, 4)

        l_p6 = self.loc_p6(p6_feats)  # (N, 24, 5, 5)
        l_p6 = l_p6.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_p6 = l_p6.view(batch_size, -1, 4)  # (N, 100, 4)

        l_p7 = self.loc_p7(p7_feats)  # (N, 16, 3, 3)
        l_p7 = l_p7.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_p7 = l_p7.view(batch_size, -1, 4)  # (N, 36, 4)

        l_p8 = self.loc_p8(p8_feats)  # (N, 16, 1, 1)
        l_p8 = l_p8.permute(0, 2, 3, 1).contiguous()  # (N, 2, 2, 16)
        l_p8 = l_p8.view(batch_size, -1, 4)  # (N, 16, 4)
        
        l_p9 = self.loc_p9(p9_feats)  # (N, 16, 1, 1)
        l_p9 = l_p9.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_p9 = l_p9.view(batch_size, -1, 4)  # (N, 4, 4)
        
        # Predict classes in localization boxes
        c_p3 = self.cl_p3(p3_feats)  # (N, 4 * n_classes, 38, 38)
        c_p3 = c_p3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_p3 = c_p3.view(batch_size, -1, self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_p4 = self.cl_p4(p4_feats)  # (N, 6 * n_classes, 19, 19)
        c_p4 = c_p4.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_p4 = c_p4.view(batch_size, -1, self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_p5 = self.cl_p5(p5_feats)  # (N, 6 * n_classes, 8, 8)
        c_p5 = c_p5.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_p5 = c_p5.view(batch_size, -1, self.n_classes)  # (N, 384, n_classes)

        c_p6 = self.cl_p6(p6_feats)  # (N, 6 * n_classes, 5, 5)
        c_p6 = c_p6.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_p6 = c_p6.view(batch_size, -1, self.n_classes)  # (N, 100, n_classes)

        c_p7 = self.cl_p7(p7_feats)  # (N, 4 * n_classes, 3, 3)
        c_p7 = c_p7.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_p7 = c_p7.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_p8 = self.cl_p8(p8_feats)  # (N, 4 * n_classes, 2, 2)
        c_p8 = c_p8.permute(0, 2, 3, 1).contiguous()  # (N, 2, 2, 4 * n_classes)
        c_p8 = c_p8.view(batch_size, -1, self.n_classes)  # (N, 16, n_classes)
        
        c_p9 = self.cl_p9(p9_feats)  # (N, 16, 1, 1)
        c_p9 = c_p9.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        c_p9 = c_p9.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # A total of 8478 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_p3, l_p4, l_p5, l_p6, l_p7, l_p8, l_p9], dim=1)  # (N, 8478, 4)
        classes_scores = torch.cat([c_p3, c_p4, c_p5, c_p6, c_p7, c_p8, c_p9], dim=1)  # (N, 8478, n_classes)

        return locs, classes_scores