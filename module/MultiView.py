# -*- coding: utf-8 -*-多视角伪目标图像处理
from typing import Dict, List, Tuple
import albumentations as A
import cv2
import torch

from dataset.base_dataset import _BaseSODDataset
from dataset.transforms.resize import ms_resize, ss_resize
from dataset.transforms.rotate import UniRotate
from dataset.transforms.GaborFliters import GaborFliter
from utils.builder import DATASETS
from utils.io.genaral import get_datasets_info_with_keys
from utils.io.image import read_color_array, read_gray_array
#import matplotlib.pyplot as plt



from PIL import Image

@DATASETS.register(name="MFFN_cod_te")
class MFFN_COD_TestDataset(_BaseSODDataset):#GSMF
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int], interp_cfg: Dict = None):
        super().__init__(base_shape=shape, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))#归一化

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        extractor = GaborFliter()

        image = read_color_array(image_path)
        image0 = cv2.flip(image, 0, dst=None)   # 作水平镜像翻转   0 表示垂直翻转，1表示水平翻转，-1表示既垂直又水平翻转
        image1 = cv2.flip(image, -1, dst=None)  # 作垂直镜像翻转
        image2 = extractor.getGabor(image)

        image = self.image_norm(image=image)["image"]#对原始图片进行一次归一化操作
        image0 = self.image_norm(image=image0)["image"]#对image0做一次归一化操作
        image1 = self.image_norm(image=image1)["image"]#对image1做一次归一化操作
        image2 = self.image_norm(image=image2)["image"]

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        image0 = ss_resize(image0, scale=1.0, base_h=base_h, base_w=base_w)#ss_resize后返回的是A.resize(img, height=int(h * scale), width=int(w * scale), interpolation=interpolation)--图像尺寸不变
        image1 = ss_resize(image1, scale=1.0, base_h=base_h, base_w=base_w)
        image2 = ss_resize(image2, scale=1.0, base_h=base_h, base_w=base_w)
        images = ms_resize(image, scales=(2.0, 1.0, 1.5), base_h=base_h, base_w=base_w)#原图像————————做扩大图像处理
        image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)  #将images转化为张量类型  permute(2, 0, 1)表示对0.1.2三个维度进行进行位置调换 第一张扩大2倍
        image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)    #将image保持原图进行张量转化
        image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)    #将image扩大1.5倍进行张量转化
        image_a_1 = torch.from_numpy(image0).permute(2, 0, 1)       #将垂直翻转图片做张量转换
        image_a_2 = torch.from_numpy(image1).permute(2, 0, 1)       #将垂直水平翻转图片做张量转换
        image_g_1 = torch.from_numpy(image2).permute(2, 0, 1)
        return dict(    #形成一个字典数据data
            data={
                "image_c1": image_0_5,  #扩大2倍张量
                "image_o": image_1_0,#原图尺寸张量
                "image_c2": image_1_5,#1.5倍张量
                "image_a1": image_a_1,#垂直翻转张量
                "image_a2": image_a_2,#垂直水平翻转张量
                "image_g1": image_g_1,
            },
            info=dict(
                mask_path=mask_path,
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)#记录数据量
@DATASETS.register(name="MFFN_cod_tr")
class MFFN_COD_TrainDataset(_BaseSODDataset):         #训练数据集处理
    def __init__(
        self, root: List[Tuple[str, dict]], shape: Dict[str, int], extra_scales: List = None, interp_cfg: Dict = None
    ):
        super().__init__(base_shape=shape, extra_scales=extra_scales, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.joint_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),    #horizontal flip水平翻转  应用变换的概率0.5
                UniRotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
        self.reszie = A.Resize
    def __getitem__(self, index):   #获取项目
        extractor = GaborFliter()
        image_path = self.total_image_paths[index]  #index--索引image中的图片
        mask_path = self.total_mask_paths[index]  #index--索引mask中的图片
        image = read_color_array(image_path)    #读取image_path给image
        image0 = cv2.flip(image, 0, dst=None)  # 作水平镜像翻转
        image1 = cv2.flip(image, -1, dst=None)  # 作垂直镜像翻转
        image2 = extractor.getGabor(image)
        mask = read_gray_array(mask_path, to_normalize=True, thr=0.5)
        transformed = self.joint_trans(image=image, mask=mask)
        transformed0 = self.joint_trans(image=image0, mask=mask)
        transformed1 = self.joint_trans(image=image1, mask=mask)
        transformed2 = self.joint_trans(image=image2, mask=mask)
        image = transformed["image"]
        image0 = transformed0["image"]
        image1 = transformed1["image"]
        image2 = transformed2["image"]
        # image0 = cv2.flip(image, 0, dst=None)  # 作水平镜像翻转
        # image1 = cv2.flip(image, -1, dst=None)  # 作垂直镜像翻转
        mask = transformed["mask"]
        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(2.0, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image0 = ss_resize(image0, scale=1.0, base_h=base_h, base_w=base_w)
        image1 = ss_resize(image1, scale=1.0, base_h=base_h, base_w=base_w)
        image2 = ss_resize(image2, scale=1.0, base_h=base_h, base_w=base_w)

        image_c_1 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_o = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_c_2 = torch.from_numpy(images[2]).permute(2, 0, 1)
        image_a_1 = torch.from_numpy(image0).permute(2, 0, 1)
        image_a_2 = torch.from_numpy(image1).permute(2, 0, 1)
        image_g_1 = torch.from_numpy(image2).permute(2, 0, 1)

        mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        mask_1_0 = torch.from_numpy(mask).unsqueeze(0)
        return dict(
            data={
                "image_c1": image_c_1,
                "image_o": image_o,
                "image_c2": image_c_2,
                "image_a1": image_a_1,
                "image_a2": image_a_2,
                "image_g1": image_g_1,
                "mask": mask_1_0,
            }
        )

    def __len__(self):
        return len(self.total_image_paths)
