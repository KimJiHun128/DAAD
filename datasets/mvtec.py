import os
from enum import Enum
import random
import torchvision.transforms as transforms
import PIL
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

def salt_noise(image, salt_prob=0.01, min_block_size=5, max_block_size=10, max_noise=4):
    # 50% 확률로 salt_noise 적용
    if random.random() < 0.5:  # 50% 확률로 salt noise 적용
        np_image = np.array(image)
        center_x, center_y = np_image.shape[0] // 2, np_image.shape[1] // 2
        start_x, start_y = center_x - 18, center_y - 18  # 중앙 36x36 영역의 시작 좌표

        for _ in range(random.randint(1, max_noise)):  # 1~4개의 노이즈를 랜덤으로 추가
            block_size = random.randint(min_block_size, max_block_size)
            x = random.randint(start_x, start_x + 36 - block_size)
            y = random.randint(start_y, start_y + 36 - block_size)
            np_image[x:x + block_size, y:y + block_size, :] = 255  # 흰색으로 설정
        return Image.fromarray(np_image)
    else:
        return image  # salt_noise를 적용하지 않으면 원본 이미지 그대로 반환


def pepper_noise(image, pepper_prob=0.01, min_block_size=5, max_block_size=10, max_noise=4):
    # 50% 확률로 pepper_noise 적용
    if random.random() < 0.5:  # 50% 확률로 pepper noise 적용
        np_image = np.array(image)
        center_x, center_y = np_image.shape[0] // 2, np_image.shape[1] // 2
        start_x, start_y = center_x - 18, center_y - 18  # 중앙 36x36 영역의 시작 좌표

        for _ in range(random.randint(1, max_noise)):  # 1~4개의 노이즈를 랜덤으로 추가
            block_size = random.randint(min_block_size, max_block_size)
            x = random.randint(start_x, start_x + 36 - block_size)
            y = random.randint(start_y, start_y + 36 - block_size)
            np_image[x:x + block_size, y:y + block_size, :] = 0  # 검은색으로 설정
        return Image.fromarray(np_image)
    else:
        return image  # pepper_noise를 적용하지 않으면 원본 이미지 그대로 반환

def apply_loma(image, p=0.5, r_min=12, r_max=18, a_min=1, a_max=3):
    """
    LOMA 데이터 증강 적용 함수 (L2 norm 기반 타원형 변형)
    :param image: 입력 이미지 (H, W, C)
    :param p: 변형 적용 확률
    :param r_min, r_max: 변형 반지름 범위
    :param a_min, a_max: 변형 강도 범위
    :return: 변형된 이미지
    """

    if np.random.rand() > p:
        return image  # 확률에 따라 변형하지 않음
    image = np.array(image)
    H, W, C = image.shape  # 이미지의 높이, 너비, 채널 수 가져오기
    xc = np.random.randint(int(W * 0.15), int(W * 0.85))  # 중심 좌표를 이미지 너비의 70% 범위 내에서 설정
    yc = np.random.randint(int(H * 0.15), int(H * 0.85))  # 중심 좌표를 이미지 높이의 70% 범위 내에서 설정
    # xc, yc = np.random.randint(W), np.random.randint(H)  # 변형 중심 좌표를 무작위로 설정
    r = np.random.randint(r_min, r_max + 1)  # 변형 반지름을 범위 내에서 무작위로 설정

    if np.random.rand() > 0.5:
        ax = np.random.uniform(a_min, a_max)  # 가로 변형 계수 (랜덤 값 할당)
        ay = 1.0  # 세로 변형 계수 (기본값 설정)
    else:
        ax = 1.0  # 가로 변형 계수 (기본값 설정)
        ay = np.random.uniform(a_min, a_max)  # 세로 변형 계수 (랜덤 값 할당)

    new_image = image.copy()  # 원본 이미지를 복사하여 변형 수행
    for i in range(H):  # 이미지 높이(H)만큼 반복
        for j in range(W):  # 이미지 너비(W)만큼 반복
            # 변형 영역에 속하는지 확인 (L2 norm 기반 타원 내 여부 판단)
            if ax * ((j - xc) ** 2) + ay * ((i - yc) ** 2) < r ** 2:
                d = np.sqrt((j - xc) ** 2 + (i - yc) ** 2)  # 픽셀과 중심 사이 거리 계산
                xo = int((d / r) * (j - xc) + xc)  # 변형 후 x 좌표 계산
                yo = int((d / r) * (i - yc) + yc)  # 변형 후 y 좌표 계산

                if 0 <= xo < W and 0 <= yo < H:  # 계산된 좌표가 이미지 범위 내에 있는지 확인
                    new_image[i, j] = image[yo, xo]  # 변형된 좌표에 기존 픽셀 값을 할당
    return Image.fromarray(new_image)


class MVTecDataset(torch.utils.data.Dataset):
    """
    CTCA dataset loader.

    The class name is kept as MVTecDataset for compatibility with the original
    SimpleNet entrypoint.
    """

    def __init__(
        self,
        source, # 데이터셋이 저장된 경로
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        fake_aug = 0,
        real_aug = 0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the CTCA data folder.
            classname: [str or list[str] or None]. CTCA patient folder(s) to
                       load. If None, the original MVTec class list is used for
                       compatibility only.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit.
        """
        super().__init__() # 부모 클래스(torch.utils.data.Dataset)의 생성자 호출
        self.source = source # CTCA 데이터 폴더 경로
        self.split = split
        self.fake_aug = fake_aug
        self.real_aug = real_aug
        if classname is None:
            self.classnames_to_use = _CLASSNAMES
        elif isinstance(classname, (list, tuple)):
            self.classnames_to_use = list(classname)
        else:
            self.classnames_to_use = [classname]
        self.train_val_split = train_val_split
        # 이미지 정규화를 위한 평균과 표준편차 설정 (ImageNet 기준)
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        # 이미지 경로 및 데이터 리스트를 가져옴
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        def get_base_transform(resize):
            return [transforms.Resize(resize)]

        def get_tensor_transform(imagesize):
            return [
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]

        def get_fake_augmentations():
            return [
                lambda x: salt_noise(x, salt_prob=0.01),
                lambda x: pepper_noise(x, pepper_prob=0.01),
                lambda x: apply_loma(x, p=0.5),
                transforms.ElasticTransform(alpha=40.0, sigma=3.0, interpolation=transforms.InterpolationMode.BILINEAR,
                                            fill=0),
            ]

        def get_real_augmentations(rotate_degrees, translate, scale, brightness_factor, contrast_factor, h_flip_p,
                                   v_flip_p):
            return [
                transforms.ColorJitter(brightness=brightness_factor, contrast=contrast_factor),
                transforms.RandomHorizontalFlip(h_flip_p),
                transforms.RandomVerticalFlip(v_flip_p),
                transforms.RandomAffine(rotate_degrees,
                                        translate=(translate, translate),
                                        scale=(1.0 - scale, 1.0 + scale),
                                        interpolation=transforms.InterpolationMode.BILINEAR),
            ]

        # if self.fake_aug ==1:
        #     base_transform = [
        #         transforms.Resize(resize),
        #     ]
        #     # fake augmentation 리스트 정의
        #     fake_augmentations = [
        #         lambda x: salt_noise(x, salt_prob=0.01),  # salt noise
        #         lambda x: pepper_noise(x, pepper_prob=0.01),  # pepper noise
        #         lambda x: apply_loma(x, p=0.5),
        #         transforms.ElasticTransform(alpha=40.0, sigma=3.0, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
        #
        #     ]
        #     # Elastic transform은 무조건 수행
        #     # 나머지 변환 리스트 (ToTensor, Normalize)
        #     tensor_transform = [
        #         transforms.CenterCrop(imagesize),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        #     ]
        #
        #     self.transform_img = base_transform + fake_augmentations + tensor_transform
        #
        #
        #
        # else:
        #     if real_aug == 1: # real aug가 1인경우 real aug 진행
        #
        #         self.transform_img = [
        #             # transforms.RandomGrayscale(gray_p),  # 랜덤 흑백 변환 (확률 gray_p)
        #             transforms.Resize(resize),
        #
        #             transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),  # 밝기, 대비, 채도 조정
        #             transforms.RandomHorizontalFlip(h_flip_p),
        #             transforms.RandomVerticalFlip(v_flip_p),
        #             transforms.RandomAffine(rotate_degrees,  # 랜덤 변환(회전, 이동, 스케일 변환 포함)
        #                                     translate=(translate, translate),
        #                                     scale=(1.0 - scale, 1.0 + scale),
        #                                     interpolation=transforms.InterpolationMode.BILINEAR),
        #
        #             transforms.CenterCrop(imagesize),  # 중앙을 기준으로 지정된 크기로 자르기
        #             transforms.ToTensor(),  # 이미지를 텐서로 변환 (픽셀 값을 [0, 1] 범위로 정규화)
        #             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # 이미지 정규화 (ImageNet 기준)
        #         ]
        #
        #     elif real_aug == 0: # real aug가 0인경우 real aug 진행 x
        #         self.transform_img = [
        #             # transforms.RandomGrayscale(gray_p),  # 랜덤 흑백 변환 (확률 gray_p)
        #             transforms.Resize(resize),
        #             transforms.CenterCrop(imagesize),  # 중앙을 기준으로 지정된 크기로 자르기
        #             transforms.ToTensor(),  # 이미지를 텐서로 변환 (픽셀 값을 [0, 1] 범위로 정규화)
        #             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # 이미지 정규화 (ImageNet 기준)
        #         ]
        #
        #
        #
        # self.transform_img = transforms.Compose(self.transform_img) # 변환들을 하나로 묶음

        self.transform_img = []

        self.transform_img += get_base_transform(resize)

        if self.fake_aug == 1:
            print("[INFO] Fake augmentation 적용 중")
            self.transform_img += get_fake_augmentations()

        if real_aug == 1:
            print("[INFO] Real augmentation 적용 중")
            self.transform_img += get_real_augmentations(rotate_degrees, translate, scale, brightness_factor,
                                                         contrast_factor, h_flip_p, v_flip_p)

        self.transform_img += get_tensor_transform(imagesize)
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        # classname, anomaly, image_path= self.data_to_iterate[idx]
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        class_basename = os.path.basename(classname)
        is_anomaly = int(class_basename.startswith("D_"))

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": is_anomaly,
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname)  # 'test' 폴더 대신 클래스 폴더를 사용
            maskpath = os.path.join(self.source, classname, "ground_truth")

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            # 여기서는 'test' 폴더가 아니라 해당 클래스 폴더 내 모든 이미지를 찾음
            image_files = sorted([f for f in os.listdir(classpath) if f.endswith(('.png', '.jpg', '.jpeg'))])

            # 모든 이미지를 `test` 이미지로 추가
            imgpaths_per_class[classname]["test"] = [os.path.join(classpath, f) for f in image_files]

            # 마스크가 필요한 경우 마스크도 추가
            if self.split == DatasetSplit.TEST:
                # 마스크가 있는 경우
                if os.path.exists(maskpath):
                    mask_files = sorted(os.listdir(maskpath))
                    maskpaths_per_class[classname]["test"] = [
                        os.path.join(maskpath, f) for f in mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["test"] = []  # 마스크가 없으면 빈 리스트

            # Unrolls the data dictionary to an easy-to-iterate list.
            data_to_iterate = []
            for classname in sorted(imgpaths_per_class.keys()):
                for anomaly in sorted(imgpaths_per_class[classname].keys()):
                    for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                        data_tuple = [classname, anomaly, image_path]
                        if self.split == DatasetSplit.TEST and anomaly != "good":
                            # 마스크가 없는 경우 None을 추가
                            if len(maskpaths_per_class[classname][anomaly]) > i:
                                data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                            else:
                                data_tuple.append(None)
                        else:
                            data_tuple.append(None)
                        data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
