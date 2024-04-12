import tensorly as tl
import numpy as np
import timm
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

from torch.utils.checkpoint import checkpoint
from Resnet_50 import ResNet50Modified
from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample



tl.set_backend('pytorch')

###############  Multi-scale features Process Module多尺度特征处理模块  ##################

class ASPP(nn.Module):  #空洞空间卷积池化金字塔(atrous spatial pyramid pooling (ASPP))对所给定的输入以不同采样率的空洞卷积并行采样。
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)#自定义函数，先进行一次卷积在归一化处理
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)      #9?
        self.conv6 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(6*out_dim, out_dim, 3, 1, 1)
    def forward(self, x):  #向前传播
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(x)
        conv6 = self.conv6(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))

        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5, conv6), 1))     #torch.cat是PyTorch张量拼接操作中的一个函数，用于在指定维度上将多个张量拼接在一起。必须包含相同形状的张量，dim(int)：默认为0。指定按哪个维度进行拼接操作。

#ASPP对原始图片进行提取，经过TransLayer形成不同尺度的特征
#对一张图片FPN后的5个尺度的特征分别进行提取保证输出通道相同，第5层卷积对参数的进行平均化，最后将5次卷积结果进行cat方式特征融合
class TransLayer(nn.Module):  #继承ASPP的TransLayer模块
    def __init__(self, out_c, last_module=ASPP):
        super().__init__()
        self.c5_down = nn.Sequential(
            # ConvBNReLU(2048, 256, 3, 1, 1),
            last_module(in_dim=512, out_dim=out_c), #输出通道定义为64时，输入通道为2048，
        )
        self.c4_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))          #输出通道相同
        self.c3_down = nn.Sequential(ConvBNReLU(128, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))
        #self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

        #__________pooling____________#



    def forward(self, xs):
        assert isinstance(xs, (tuple, list))

        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs

        #c6 = self.c6_down(c6)
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)

        #print("c1",c1)
        return  c5, c4, c3, c2, c1   #分别表示主干网络中不同尺度的特征图，返回6个尺度。


###############  Cross-View Attention Module  ##################


class ChannelAttention(nn.Module):  #---------------通道注意力机制 PoolAttention
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     #输出特征图大小
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)  #其中1为卷积核大小
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  #平均池化的输出
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  #最大池化处理的输出
        out = avg_out + max_out #平均池化和最大池化的Add
        return self.sigmoid(out)
#运算结果为返回为经过sigmod函数的值
class SpatialAttention(nn.Module):       #空间注意力
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)   #输入输出通道
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    #对x张量求每行的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  #对x张量求每列的最大值
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#—————————————————————————————heatmap池化注意力————————————————————————————————#
"""class PoolingAttention(nn.Module):                      #需要修改维度和通道
    def __int__(self, in_dim):
        super(PoolingAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 1, 1, 0)
        self.Avg_pool1 = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Conv2d(in_dim, in_dim, 1, 1, 0)
        self.Max_pool2 = nn.AdaptiveMaxPool2d(1)

        self.avg_pool3 = nn.AdaptiveAvgPool2d(1)  # 输出特征图大小
        self.max_pool3 = nn.AdaptiveMaxPool2d(1)

        self.conv_g = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
            ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        """

class FeatureNorm(nn.Module):
    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(2, 3), eps=0.001, include_bias=True):
        super(FeatureNorm, self).__init__()
        self.shape = [1] * rank
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims

        self.scale = nn.Parameter(torch.ones(self.shape, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(self.shape, requires_grad=True, dtype=torch.float)) if include_bias else nn.Parameter(
            torch.zeros(self.shape, requires_grad=False, dtype=torch.float))

        self.eps = eps

    def forward(self, features):
        f_std = torch.std(features, dim=self.reduce_dims, keepdim=True)
        f_mean = torch.mean(features, dim=self.reduce_dims, keepdim=True)
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()) + self.bias



class Conv2d_init(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super(Conv2d_init, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)




def _conv_block(in_chanels, out_chanels, kernel_size, padding):
    return nn.Sequential(Conv2d_init(in_channels=in_chanels, out_channels=out_chanels,
                                     kernel_size=kernel_size, padding=padding, bias=False),
                         FeatureNorm(num_features=out_chanels, eps=0.001),
                         nn.ReLU())



class heatmap(nn.Module):
    def __init__(self, input_channels):
        super(heatmap, self).__init__()
        self.input_channels = input_channels
        self.heatmappooling1 = nn.Sequential(
            _conv_block(self.input_channels, 64, 5, padding=2),
            _conv_block(64, 8, 5, padding=2),            # Has been accidentally left out and remained the same since then被意外地排除在外，然后一直保持不变
            nn.MaxPool2d(2,stride=2),
        )
        self.heatmappooling2 = nn.Sequential(
             _conv_block(256, 64, 5,padding=2),
             _conv_block(64, 64, 5, padding=2),
             _conv_block(64, 8, 5, padding=2),
             nn.MaxPool2d(2,stride=2),  # 池化不改变通道数

         )
        self.heatmappooling3 = nn.Sequential(
                                                 _conv_block(128, 64, 5, padding=2),
                                                 _conv_block(64, 64, 5, padding=2),
                                                 _conv_block(64, 64, 5, padding=2),
                                                 _conv_block(64, 8, 5, padding=2),
                                                 nn.MaxPool2d(2,stride=2),
        )            # External_attention(64),
        self.heatmappooling4 = nn.Sequential(
                _conv_block(64,8, 5, padding=2),
                nn.MaxPool2d(2, stride=2),
        )
        self.global_max_pool_feat = nn.AdaptiveMaxPool2d((384,384))
        self.global_avg_pool_feat = nn.AdaptiveAvgPool2d((384,384))
    def forward(self, xs):
        assert isinstance(xs, (tuple, list))

        assert len(xs) ==5
        c0, c1, c2, c3, c4= xs
        """ print(".................",c0.shape)
        print(c1.shape)
        print(c2.shape)
        print(c3.shape)
        print(c4.shape)
        #print(c5.shape)"""


        c4 = self.heatmappooling1(c4)
        c41 = self.global_avg_pool_feat(c4)
        c42 = self.global_max_pool_feat(c4)
        c3 = self.heatmappooling2(c3)
        c31 = self.global_avg_pool_feat(c3)
        c32 = self.global_max_pool_feat(c3)
        c2 = self.heatmappooling3(c2)
        c21 = self.global_avg_pool_feat(c2)
        c22 = self.global_max_pool_feat(c2)
        c1 = self.heatmappooling4(c1)
        c11 = self.global_avg_pool_feat(c1)
        c12 = self.global_max_pool_feat(c1)

        heatmap_valum = torch.cat([c41, c42, c31, c32, c21, c22, c11, c12], dim=1)
       # print("heatmap",heatmap_valum.shape)
        return  heatmap_valum  # 分别表示主干网络中不同尺度的特征图，返回4个尺度。

class CAMV(nn.Module):   #全局多视角注意力机制-------------------------------------需要加入第六层
    def __init__(self, in_dim, mm_size):
        super().__init__()
        #self.Pool=PoolingAttention()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)   #自定义函数，先进行一次卷积在归一化处理  post process处理过程    #1
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)                                                                          #2
        self.conv_m = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            #nn.ReLU(),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )                                                                                                                                    #3原图
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)                                                                             #4
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)                                                                            #5


        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
            nn.ReLU(),
            ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
            nn.ReLU(),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.trans1 = nn.Sequential(
            ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
            ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
       
        self.transa1 = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.transa2 = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.transg = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )


        self.transg1 = nn.Sequential(
            ConvBNReLU(in_dim, 3 * in_dim, 1),
            nn.ReLU(),
            ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            )


        self.transg1_ =nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 1),
            nn.ReLU(),
                                     )




        self.mm_size = mm_size
        self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)   #PyTorch中的一个类，用于将一个Tensor对象封装为一个可训练的参数，只需要传入一个Tensor对象即可
        self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)    #把3输入的张量大小转换为相同大小
        self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_c1.data.uniform_(-0.5,0.5)   #随机生成-0.5，0.5之间的一个数，将data的值进行调整
        self.coe_h_c1.data.uniform_(-0.5,0.5)
        self.coe_w_c1.data.uniform_(-0.5,0.5)
        
        self.coe_c_md.data.uniform_(-0.5,0.5)
        self.coe_h_md.data.uniform_(-0.5,0.5)
        self.coe_w_md.data.uniform_(-0.5,0.5)
        
        self.coe_c_c2.data.uniform_(-0.5,0.5)
        self.coe_h_c2.data.uniform_(-0.5,0.5)
        self.coe_w_c2.data.uniform_(-0.5,0.5)
        
        self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_a1.data.uniform_(-0.5,0.5)#会把coe_c_a1 中的data=torch.Tensor(1,64)中的1，64重新赋值
        self.coe_h_a1.data.uniform_(-0.5,0.5)#把coe_h_a1中的data=torch.Tensor(mm_size,mm_size)中mm_size,mm_size进行重新赋值。
        self.coe_w_a1.data.uniform_(-0.5,0.5)
        
        self.coe_c_ma.data.uniform_(-0.5,0.5)
        self.coe_h_ma.data.uniform_(-0.5,0.5)
        self.coe_w_ma.data.uniform_(-0.5,0.5)
        
        self.coe_c_a2.data.uniform_(-0.5,0.5)
        self.coe_h_a2.data.uniform_(-0.5,0.5)
        self.coe_w_a2.data.uniform_(-0.5,0.5)
       # self.pool_attn = PoolingAttention()
        self.channel_attn = ChannelAttention(64)    #通道注意力
        self.spatial_attn = SpatialAttention()  #空间注意力
        self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
    def forward(self, c1, o, c2, a1, a2, g1, return_feats=False):
        tgt_size = o.shape[2:]       #获得张量尺寸信息获得o的高H和宽W信息   #“o”张量（也称为tensor）的第二个维度以及之后的所有维度切片，并将结果赋值给“tgt_size”变量。就是W--宽

        c1 = self.conv_l_pre_down(c1)  #下采样计算过程

        c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)  #adaptive_max_pool2d(c1, tgt_size)输入尺寸输出尺寸
        c1 = self.conv_l_post_down(c1)
        g1_pooling = g1


        g1 = self.conv_l_pre_down(g1)
        g1 = F.adaptive_max_pool2d(g1, tgt_size) + F.adaptive_avg_pool2d(g1, tgt_size)
        g1 = self.conv_l_post_down(g1)


        #m = self.conv_m(o)
        #print("m:",m.shape)

        # g1 = self.conv_m(g1)

        c2 = self.conv_s_pre_up(c2)
        c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
        c2 = self.conv_s_post_up(c2)
        attn = self.trans(torch.cat([c1, g1, c2], dim=1))        #CAMV的第一阶段特征图融合attn: torch.Size([1, 64, 12, 12])

        g1_ = self.transg1(g1_pooling)
        g1__ = F.adaptive_max_pool2d(g1_, tgt_size)
        g1___ = F.adaptive_avg_pool2d(g1_,tgt_size)
        """print("g1_",g1_.shape)
        print("g1__", g1_.shape)
        print("g1___", g1_.shape)"""
        #g1_pooling_cat =torch.cat([g1_, g1__, g1___], dim=1)
        #print("torch.cat([g1_, g1__, g1___]", torch.cat([g1_, g1__, g1___],dim=1).shape)
        attn_c1 = tl.tenalg.mode_dot(attn, self.coe_c_c1, mode=1)  #张量乘矩阵

        attn_c1 = tl.tenalg.mode_dot(attn_c1, self.coe_h_c1, mode=2)    #一个Ua的过程
        attn_c1 = tl.tenalg.mode_dot(attn_c1, self.coe_w_c1, mode=3)
        attn_c1 = torch.softmax(attn_c1, dim=1) #当dim=0时， 是对每一维度相同位置的数值进行softmax运算，和为1
                                                #当dim=1时， 是对某一维度的列进行softmax运算，和为1
                                                #当dim=2时， 是对某一维度的行进行softmax运算，和为1
        
        attn_md = tl.tenalg.mode_dot(attn, self.coe_c_md, mode=1)
        attn_md = tl.tenalg.mode_dot(attn_md, self.coe_h_md, mode=2)
        attn_md = tl.tenalg.mode_dot(attn_md, self.coe_w_md, mode=3)
        attn_md = torch.softmax(attn_md, dim=1)
        
        attn_c2 = tl.tenalg.mode_dot(attn, self.coe_c_c2, mode=1)
        attn_c2 = tl.tenalg.mode_dot(attn_c2, self.coe_h_c2, mode=2)
        attn_c2 = tl.tenalg.mode_dot(attn_c2, self.coe_w_c2, mode=3)
        attn_c2 = torch.softmax(attn_c2, dim=1)

        attn_c2 = torch.softmax(attn_c1 + attn_c2,dim=1)
        cmc = attn_c1 * c1 + attn_md * g1 + attn_c2 * c2        #Fi-Dist



        a1 = self.transa1(a1)           #第二阶段
        a2 = self.transa2(a2)
        m = self.conv_m(o)

       # print("torch.cat([a1, m, a2]",torch.cat([a1, m, a2],dim=1).shape)
        #print("torch.cat([a1, m, a2], dim=1) + torch.cat([g1_, g1__, g1___], dim=1 )", (torch.cat([a1, m, a2], dim=1) + torch.cat([g1_, g1__, g1___], dim=1 )).shape)
        attn1 = self.trans1(torch.cat([a1, m, a2], dim=1) + torch.cat([g1_, g1__, g1___], dim=1 ))
        #print("attn1",attn1.shape)


        attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)

        attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
        attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
        attn_a1 = torch.softmax(attn_a1, dim=1)
        
        attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
        attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
        attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
        attn_ma = torch.softmax(attn_ma, dim=1)
        
        attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
        attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
        attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
        attn_a2 = torch.softmax(attn_a2, dim=1)


        attn_a2 = torch.softmax(attn_a1 + attn_a2,dim=1)
        ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2       #Fi-Ang


        ama = ama.mul(self.channel_attn(ama))
        ama = ama.mul(self.spatial_attn(ama))
        lms = self.fuse(torch.cat([ama,cmc],dim=1))
        #print("00000000000000000000000000000000000000000",lms.shape)
        return lms  #输出的还是5个--更改



class Progressive_Iteration(nn.Module): #The overall progressive iteration整体渐进迭代 OPI
    def __init__(self, input_channels):
        super(Progressive_Iteration, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)
        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x) #X为Zi
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)
        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)
        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
        return ce

class CFU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups
        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups                #512
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)
        self.fp = Progressive_Iteration(192)


    def forward(self, x):   #X为一个CAMV的输出fi
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)          #文章中定义了四个块 ,维度选择行
        outs = []
        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(2, dim=1))


        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(2, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(1, dim=1))
        out = torch.cat([o[0] for o in outs], dim=1)
        out = self.fp(out)
        out = self.fuse(out)
        return self.final_relu(out + x)

def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.5, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)

    return loss_map.mean()


@MODELS.register()
class MFFN(BasicModelClass):
    def __init__(self):
        super().__init__()
        #self.shared_encoder = ResNet50Modified()
        self.shared_encoder = timm.create_model(model_name="resnet18", pretrained=True, in_chans=3, features_only=True)
        #resnet50是五层所有有问题
        #这行代码的作用是创建一个使用ResNet50架构的预训练模型，并且仅返回一个特征提取器（即仅返回特征层，而不是全连接层）。
        # 其中，使用了timm工具包中的create_model函数，可以方便地创建常见的计算机视觉模型。ResNet50是一种适用于图像分类和目标检测等任务的深度卷积神经网络，通过残差连接的方式解决了深度卷积神经网络中的梯度消失问题。
        # 在这个模型中，输入图像的通道数为3。特征提取器会输出特征图，可以用于后续的任务，如分类、目标检测等。由于使用了预训练模型，因此可以大大缩短模型的训练时间，并提高模型的泛化能力。
        self.translayer = TransLayer(out_c=64)  # [c6, c5, c4, c3, c2, c1] #对参数进行整合提取每次尺度的张量
        dim = [64, 64, 64, 64, 64]
        size = [12,24,48,96,192]
        self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])


        #self.input_channels=3
        self.heatmap_layer = heatmap(input_channels=512)

        # self.d6 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))

        self.g1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))


        self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(32, 1, 1)







    def encoder_translayer(self, x):
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)
        return trans_feats  #返回6个尺度的特征

    def heatmap_encoder(self, x):
        en_heatmap_feat = self.shared_encoder(x)
        heat_encoder1 = self.heatmap_layer(en_heatmap_feat)
        return heat_encoder1


    def body(self,  g1_scale, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):    #注意力层的输出fi
        c1_trans_feats = self.encoder_translayer(c1_scale)   #每一个输出在经过一次特征提取
        o_trans_feats = self.encoder_translayer(o_scale)
        c2_trans_feats = self.encoder_translayer(c2_scale)
        a1_trans_feats = self.encoder_translayer(a1_scale)
        a2_trans_feats = self.encoder_translayer(a2_scale)
        g1_trans_feats = self.encoder_translayer(g1_scale)
        feats = []
        g1_heatmap_feats = self.heatmap_encoder(g1_scale)       #[192,64,1,1]


        #g1_heatmap_feat = self.heatmap_encoder(g1_scale)


        for  c1, o, c2, a1, a2, g1, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, g1_trans_feats, self.CAMV_layers):   #预测进行尺度提取
            CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2, g1=g1)
            feats.append(CAMV_outs)     #输出5个f1、f2、 f3 、f4 、f5



        """
        print(feats[0].shape)
        print(feats[1].shape)
        print(feats[2].shape)
        print(feats[3].shape)
        print(feats[4].shape)
        # print(feats[5].shape)"""


        x = self.d5(feats[0])
        #print("d6",x.shape)
        x = cus_sample(x, mode="scale", factors=2)  #x: torch.Size([1, 64, 48, 48])
        #print("x",x.shape)
        x = self.d4(x + feats[1])
        #print("d5", x.shape)
        x = cus_sample(x, mode="scale", factors=2)
        #print("x", x.shape)
        x = self.d3(x + feats[2])
        #print("d4", x.shape)
        x = cus_sample(x, mode="scale", factors=2)
        #print("x", x.shape)
        x = self.d2(x + feats[3])   #96
        #print("d3", x.shape)
        x = cus_sample(x, mode="scale", factors=2)
        #print("x", x.shape)
        x = self.d1(x + feats[4])   #192
        #print("d2", x.shape)
        x = cus_sample(x, mode="scale", factors=2)
        #print("x", x.shape)
        #print(feats[5].shape)
        #x = self.d1(x + feats[5]) #384
        #print("d1", x.shape)
        #x = cus_sample(x, mode="scale", factors=2)
        #print(x.shape)

        #print("g1_heatmap_feats",g1_heatmap_feats.shape)
        #print("1",g1_heatmap_feats.shape)
        g = self.g1(g1_heatmap_feats)
        #print("00.0.0.0.",g.shape)
        #print("2",g.shape)
        #print("x",x.shape)
        x = x + g
        #x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        #print(logits.shape)
        return dict(seg=logits)
    def train_forward(self, data, **kwargs):
        assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "image_g1", "mask"}.difference(set(data)), set(data)
        output = self.body(
            c1_scale=data["image_c1"],
            o_scale=data["image_o"],
            c2_scale=data["image_c2"],
            a1_scale=data["image_a1"],
            a2_scale=data["image_a2"],
            g1_scale=data["image_g1"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str


    def test_forward(self, data, **kwargs):
        output = self.body(
            c1_scale=data["image_c1"],
            o_scale=data["image_o"],
            c2_scale=data["image_c2"],
            a1_scale=data["image_a1"],
            a2_scale=data["image_a2"],
            g1_scale=data["image_g1"],
        )
        return output["seg"]    ##################?
    """
    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        ual_coef = get_coef(iter_percentage, method)
        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():   #通过迭代 all_preds.items()，可以依次获取每个键值对，将键存储在 name 变量中，将对应的值存储在 preds 变量中
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
            sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")    #用于计算二分类交叉熵损失（loss）的函数
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")       #mane预测值
            ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            ual_loss *= ual_coef
            losses.append(ual_loss)
            loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
        return sum(losses), " ".join(loss_str)

    """


    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        ual_coef = get_coef(iter_percentage, method)
        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
            #bce_loss = nn.BCEWithLogitsLoss()
            #sod_loss = bce_loss(preds, resized_gts)  # 使用 nn.BCEWithLogitsLoss 损失函数
            sod_loss=F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
            #ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            #ual_loss *= ual_coef
            #losses.append(ual_loss)
            #loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
        return  sum(losses), " ".join(loss_str)




    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("shared_encoder.layer"):
                param_groups.setdefault("pretrained", []).append(param)
            elif name.startswith("shared_encoder."):
                param_groups.setdefault("fixed", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        return param_groups


