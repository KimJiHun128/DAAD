# ------------------------------------------------------------------
# DAAD: CTCA anomaly detection experiments built on the SimpleNet codebase.
# Original SimpleNet source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

"""detection methods."""
import logging
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import common
import metrics
from utils import plot_segmentation_images
import pickle
import itertools
import gc
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



LOGGER = logging.getLogger(__name__)


def plot_image(tensor, title):
    # tensor가 1D일 경우, 이를 2D로 변환
    if len(tensor.shape) == 1:
        # 예: 256 크기의 벡터를 16x16 이미지로 변환
        img = tensor.detach().cpu().numpy().reshape(16, 16)
    else:
        # tensor가 이미 2D 또는 3D 형태라면 그대로 사용
        img = tensor.detach().cpu().numpy()
    # 최소값, 최대값을 기준으로 이미지를 재조정
    img_min = img.min()
    img_max = img.max()
    img_contrast = (img - img_min) / (img_max - img_min)  # [0, 1] 범위로 정규화

    plt.imshow(img_contrast, cmap='viridis')  # 이미지 출력, grayscale로 표시
    plt.title(title)
    plt.axis('off')  # 축을 제거
    plt.show()

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1

class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device # GPU 설정
        self.use_data_parallel = True
    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=6,
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        true_FLA = 1,
        false_FLA=1,
        fs_keep_ratio = 0.5,
        fs_mode = "both",
        image_gap = 5,
        **kwargs,
    ):
        pid = os.getpid()
        def show_mem():
            return(psutil.Process(pid).memory_info())
        self.true_FLA = true_FLA
        self.false_FLA = false_FLA
        self.fs_keep_ratio = fs_keep_ratio
        self.fs_mode = fs_mode
        self.image_gap = image_gap


        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({}) # forward_modules를 빈 ModuleDict로 초기화

        # 백본에서 특징을 추출  feature aggregator 초기화
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape) # 특징 차원 계산
        self.forward_modules["feature_aggregator"] = feature_aggregator # feature_aggregator를 forward_modules에 저장
        # 전처리 모듈
        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension # 추출된 특징 차원 # 사전 학습된 임베딩 차원
        )
        self.forward_modules["preprocessing"] = preprocessing # preprocessing 모듈을 forward_modules에 저장
        # 목표 임베딩 차원 설정
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator( # preadapt_aggregator 초기화
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)# preadapt_aggregator를 지정된 장치로 이동

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator # preadapt_aggregator를 forward_modules에 저장

        self.anomaly_segmentor = common.RescaleSegmentor( # 이상치 감지를 위한 RescaleSegmentor 초기화
            device=self.device, target_size=input_shape[-2:] # 리스케일할 대상 크기 (입력의 마지막 두 차원)
        )
        # 임베딩 크기를 설정. 주어진 embedding_size가 없으면 target_embed_dimension을 사용
        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone: # 만약 백본을 학습할 경우, 백본 네트워크의 옵티마이저를 설정
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj # pre_proj 값 설정

        if self.pre_proj > 0: # pre_proj 값이 0보다 크면, 사전 프로젝션을 설정
            # Projection 모듈을 초기화: 입력과 출력 차원은 target_embed_dimension으로 설정
            # pre_proj는 프로젝션의 레이어 수 또는 어떤 방식으로 프로젝션을 적용할지를 결정하는 파라미터
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr*.1) # pre_projection을 학습하는 옵티마이저를 설정: AdamW 사용, 학습률을 기본 값의 0.1로 설정

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin= dsc_margin 

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)

    def apply_feature_augmentation(self, feats, case, patch_gap, image_gap, lambda_value=0.5):
        """
        주어진 특징 맵(feats)에 대해 선택된 모드(case)에 따라 feature-level augmentation을 적용합니다.

        Parameters:
        - feats (Tensor): 입력 특징 맵 (N, C).
        - case (int): augmentation 모드
            - 0: 아무 처리 안 함
            - 1: 노이즈 추가
            - 2: 내삽 (interpolation)
            - 3: 외삽 (extrapolation)
            - 5: 내삽 + 외삽
            - 6: 노이즈 + 내삽 + 외삽
        - patch_gap (int): 특징 간 간격을 지정하는 값.
        - image_gap (int): 이미지 간 간격을 지정하는 값.
        - lambda_value (float): 내삽/외삽 시 조절 계수 (기본값: 0.5).

        Returns:
        - final_feats (Tensor): 증강이 적용된 특징 맵.
        """

        # 그대로 유지
        if case == 0:
            final_feats = feats

        # 노이즈 추가
        elif case == 1:
            with torch.no_grad():
                noise_idxs = torch.randint(0, self.mix_noise, (feats.shape[0],), device=self.device)  # (N,)
                noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(
                    self.device)  # (N, K)
                noise_list = [
                    torch.normal(mean=0, std=self.noise_std * (1.1 ** k), size=feats.shape, device=self.device)
                    for k in range(self.mix_noise)
                ]
                noise = torch.stack(noise_list, dim=1)  # (N, K, C)
                noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)  # (N, C)



            # 노이즈 추가는 그래프에 포함되어야 하므로 여기서는 detach 하지 않음
            final_feats = feats + noise
            del noise_idxs, noise_one_hot, noise


            # noise_idxs = torch.randint(0, self.mix_noise,torch.Size([feats.shape[0]]))  # (N,) 크기의 텐서를 생성하여 각 샘플마다 사용할 노이즈 레벨을 랜덤하게 선택
            # noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device)  # (N, K) 크기의 원-핫 벡터 생성 (N개의 샘플, K개의 노이즈 레벨)
            #
            # # 여러 개의 노이즈를 생성하여 스택
            # noise = torch.stack([
            #     torch.normal(0, self.noise_std * 1.1 ** (k), feats.shape)  # 노이즈 표준편차를 1.1^k 배로 증가
            #     for k in range(self.mix_noise)], dim=1).to(self.device)  # (N, K, C) # (N, K, C) 크기의 노이즈 텐서 생성 (N개의 샘플, K개의 노이즈 레벨, C개의 특징 차원)
            #
            # # 각 샘플에 대해 선택된 노이즈만 적용
            # noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)  # (N, C) 크기로 변환 - 샘플별 선택된 노이즈만 남도록 합산
            #
            # # 참 특징에 노이즈를 추가하여 가짜 특징 생성
            # final_feats = feats + noise

        # 내삽
        elif case == 2:
            interpolated_feats = torch.empty_like(feats) # feats를 복사하여 사용
            for i in range(feats.shape[0] - patch_gap * image_gap):
                if i + patch_gap * image_gap < feats.shape[0]:
                    feat1 = feats[i]
                    feat2 = feats[i + patch_gap * image_gap]
                    interpolated_feat = (1 - lambda_value) * feat1 + lambda_value * feat2  # 내삽 공식
                    interpolated_feats[i] = interpolated_feat

            # 업데이트되지 않은 부분을 제외한 interpolated_feats 업데이트된 부분만 선택
            updated_interpolated_feats = interpolated_feats[:feats.shape[0] - patch_gap * image_gap]
            # feats와 updated_interpolated_feats를 연결
            final_feats = torch.cat((feats, updated_interpolated_feats), dim=0)

        # 외삽
        elif case == 3:
            extrapolated_feats = torch.empty_like(feats)  # feats를 복사하여 사용
            for i in range(feats.shape[0] - patch_gap * image_gap):
                if i + patch_gap * image_gap < feats.shape[0]:
                    feat1 = feats[i]
                    feat2 = feats[i + patch_gap * image_gap]
                    extrapolated_feat = (1 + lambda_value) * feat1 - lambda_value * feat2
                    extrapolated_feats[i] = extrapolated_feat

            # 업데이트되지 않은 부분을 제외한 extrapolated_feats 업데이트된 부분만 선택
            updated_extrapolated_feats = extrapolated_feats[:feats.shape[0] - patch_gap * image_gap]
            final_feats = torch.cat((feats, updated_extrapolated_feats), dim=0)

        # 내삽 + 외삽
        elif case == 5:
            interpolated_feats = torch.empty_like(feats)
            extrapolated_feats = torch.empty_like(feats)

            for i in range(feats.shape[0] - patch_gap * image_gap):
                if i + patch_gap * image_gap < feats.shape[0]:
                    feat1 = feats[i]
                    feat2 = feats[i + patch_gap * image_gap]

                    # 내삽
                    interpolated_feat = (1 - lambda_value) * feat1 + lambda_value * feat2
                    interpolated_feats[i] = interpolated_feat

                    # 외삽
                    extrapolated_feat = (1 + lambda_value) * feat1 - lambda_value * feat2
                    extrapolated_feats[i] = extrapolated_feat

                    # if i == 0:  # 첫 번째 쌍에 대해서만 시각화
                        # plot_image(feat1, "Feature 1")
                        # plot_image(feat2, "Feature 2")
                        # plot_image(interpolated_feat, "Interpolated Feature")
                        # plot_image(extrapolated_feat, "Extrapolated Feature")

                        # # 1D 벡터로 변환
                        # f1 = feat1.detach().cpu().numpy().flatten()
                        # f2 = feat2.detach().cpu().numpy().flatten()
                        # inter = interpolated_feat.detach().cpu().numpy().flatten()
                        # extra = extrapolated_feat.detach().cpu().numpy().flatten()
                        #
                        # x = np.arange(len(f1))  # index를 x축으로 사용

                        # plt.figure(figsize=(6, 4))
                        # plt.plot(x[:25], f1[:25], label="Feature 1", color="blue")
                        # plt.plot(x[:25], f2[:25], label="Feature 2", color="red")
                        # plt.plot(x[:25], inter[:25], label="Interpolated", color="purple", linestyle='--')
                        # plt.plot(x[:25], extra[:25], label="Extrapolated", color="orange", linestyle='--')
                        # plt.legend()
                        # plt.title("Feature Interpolation & Extrapolation (Curve View)")
                        # plt.xlabel("Index")
                        # plt.ylabel("Feature Value")
                        # plt.grid(True)
                        # plt.tight_layout()
                        # plt.show()


            updated_interpolated_feats = interpolated_feats[:feats.shape[0] - patch_gap * image_gap]
            updated_extrapolated_feats = extrapolated_feats[:feats.shape[0] - patch_gap * image_gap]

            # # 원본 feats와 내삽, 외삽 결과를 모두 합치기
            # final_feats = torch.cat((feats, updated_interpolated_feats, updated_extrapolated_feats), dim=0)

            # 내삽 + 외삽만 합치기
            final_feats = torch.cat((updated_interpolated_feats, updated_extrapolated_feats), dim=0)

            # 노이즈 + 내삽 + 외삽
        elif case == 6:
            # ---------------------
            # 1. 노이즈 추가

            noise_idxs = torch.randint(0, self.mix_noise, (feats.shape[0],), device=self.device)
            noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device)
            noise = torch.stack([
                torch.normal(0, self.noise_std * 1.1 ** k, feats.shape, device=self.device)
                for k in range(self.mix_noise)
            ], dim=1)
            noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
            noise_feats = feats + noise

            # noise_idxs = torch.randint(0, self.mix_noise,torch.Size([feats.shape[0]]))
            # noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device)
            # noise = torch.stack([
            #     torch.normal(0, self.noise_std * 1.1 ** (k), feats.shape)
            #     for k in range(self.mix_noise)], dim=1).to(self.device)
            # noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
            # noise_feats = feats + noise

            # ---------------------
            # 2. 내삽
            interpolated_feats = torch.empty_like(noise_feats)
            for i in range(noise_feats.shape[0] - patch_gap * image_gap):
                if i + patch_gap * image_gap < noise_feats.shape[0]:
                    feat1 = noise_feats[i]
                    feat2 = noise_feats[i + patch_gap * image_gap]
                    interpolated_feats[i] = (1 - lambda_value) * feat1 + lambda_value * feat2
            interpolated_feats = interpolated_feats[:noise_feats.shape[0] - patch_gap * image_gap]

            # ---------------------
            # 3. 외삽
            extrapolated_feats = torch.empty_like(noise_feats)
            for i in range(noise_feats.shape[0] - patch_gap * image_gap):
                if i + patch_gap * image_gap < noise_feats.shape[0]:
                    feat1 = noise_feats[i]
                    feat2 = noise_feats[i + patch_gap * image_gap]
                    extrapolated_feats[i] = (1 + lambda_value) * feat1 - lambda_value * feat2
            extrapolated_feats = extrapolated_feats[:noise_feats.shape[0] - patch_gap * image_gap]

            # ---------------------
            # 모두 합치기
            final_feats = torch.cat((interpolated_feats, extrapolated_feats), dim=0)

            del noise_idxs, noise_one_hot, noise




        return final_feats

    def apply_feature_selection(self, feats: torch.Tensor, keep_ratio: float = 0.5, mode: str = 'both'):
        """
        feature tensor [N, C]에서 GAP/GMP 수행 후 top-k feature map 선택

        Args:
            feats (Tensor): [N, C] 형태 (N: feature map 개수, C: feature vector 크기)
            keep_ratio (float): 남길 feature map 비율 (0~1)
            mode (str): 'gap', 'gmp', 'both' 중 하나

        Returns:
            selected_feats (Tensor): [k, C] 선택된 feature map만 남긴 텐서
        """
        N, C = feats.shape
        k = max(1, int(N * keep_ratio))

        if mode == 'gap':
            pooled = feats.mean(dim=1)  # [N]
            _, topk_idx = torch.topk(pooled, k=k) # indices extract
            selected_feats = feats[topk_idx, :] # [k, C]
        elif mode == 'gmp':
            pooled = feats.amax(dim=1)  # [N]
            _, topk_idx = torch.topk(pooled, k=k)
            selected_feats = feats[topk_idx, :]
        elif mode == 'both':
            pooled_gap = feats.mean(dim=1)
            pooled_gmp = feats.amax(dim=1)

            _, topk_idx_gap = torch.topk(pooled_gap, k=k)
            _, topk_idx_gmp = torch.topk(pooled_gmp, k=k)

            topk_idx = torch.unique(torch.cat([topk_idx_gap, topk_idx_gmp], dim=0))
            selected_feats = feats[topk_idx, :]

        elif mode == 'none':
            selected_feats = feats
        else:
            raise ValueError("mode는 'gap', 'gmp', 'both', 'none' 중 하나여야 합니다.")

        return selected_feats


    def embed(self, data): #image 데이터를 추출하고, input_image로 변환한 후 _embed를 호출
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():# 학습을 하지 않고 특징을 추출 (gradient 계산 없이)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images.""" # 실제로 이미지의 특징 벡터를 추출하는 함수
        #  이미지를 통과시켜 특징을 뽑아낸 뒤, 여러 레이어에서 특징을 추출하여 크기를 조정하고, patchify를 적용한 후 다양한 전처리 과정을 진행하여 최종 특징 벡터를 반환
        B = len(images)
        # 학습 중이 아니고 평가 모드일 경우, feature_aggregator 모듈을 eval 모드로 전환
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else: # 평가 모드일 경우 feature_aggregator를 eval로 설정하고 특징 추출
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)
        # 지정된 레이어에서 추출한 특징들만 가져오기
        features = [features[layer] for layer in self.layers_to_extract_from]
        # 특징의 차원을 변경하여 이미지 형태로 변환 128,5,5 256,3,3
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                print("B, L, C: ", B,L,C)
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
        # 각 특징에 대해 patchify 함수 적용 (이미지를 패치로 분할)
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        # 패치 정보와 특징을 분리
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        # 첫 번째 특징의 패치 수를 기준으로 맞추기
        ref_num_patches = patch_shapes[0]
        # 나머지 특징들을 첫 번째 특징에 맞게 크기 변경
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            # 특징 크기를 재조정
            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )

            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features # 특징을 리스트에 다시 저장
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]# 모든 특징을 플랫하게 펼침
        # 다양한 네트워크 백본과 패치 처리 방식에 맞춰 특징들의 크기를 동일하게 맞춤
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling

        # import matplotlib.pyplot as plt
        # import numpy as np
        #
        # # 첫 번째 배치 항목을 가져옵니다.
        # first_feature = features[0].cpu().numpy()
        #
        # # 256 크기를 16x16 이미지로 변환
        # image = first_feature.reshape(16, 16)
        #
        # # 이미지를 시각화합니다.
        # plt.imshow(image, cmap='viridis')  # cmap을 원하는 색상맵으로 변경 가능
        # plt.colorbar()  # 색상바 표시
        # plt.show()
        #
        return features, patch_shapes






    # def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
    #
    #     scores = np.squeeze(np.array(scores))
    #     img_min_scores = scores.min(axis=-1)
    #     img_max_scores = scores.max(axis=-1)
    #     scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
    #     # scores = np.mean(scores, axis=0)
    #
    #     auroc = metrics.compute_imagewise_retrieval_metrics(
    #         scores, labels_gt
    #     )["auroc"]
    #
    #     if len(masks_gt) > 0:
    #         segmentations = np.array(segmentations)
    #         min_scores = (
    #             segmentations.reshape(len(segmentations), -1)
    #             .min(axis=-1)
    #             .reshape(-1, 1, 1, 1)
    #         )
    #         max_scores = (
    #             segmentations.reshape(len(segmentations), -1)
    #             .max(axis=-1)
    #             .reshape(-1, 1, 1, 1)
    #         )
    #         norm_segmentations = np.zeros_like(segmentations)
    #         for min_score, max_score in zip(min_scores, max_scores):
    #             norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
    #         norm_segmentations = norm_segmentations / len(scores)
    #
    #
    #         # Compute PRO score & PW Auroc for all images
    #         pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
    #             norm_segmentations, masks_gt)
    #             # segmentations, masks_gt
    #         full_pixel_auroc = pixel_scores["auroc"]
    #
    #         pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)),
    #                                         norm_segmentations)
    #     else:
    #         full_pixel_auroc = -1
    #         pro = -1
    #
    #     return auroc, full_pixel_auroc, pro

    def plot_scores(self, scores, save_path, dataset_name="Unknown Dataset"):
        """
        scores 값을 하나의 배열로 합쳐 시각화하고, 이미지로 저장하는 함수.
        """

        # 0부터 -1까지 0.1 단위로 범위 설정
        bins = np.arange(-1, 1.1, 0.1)
        counts, bin_edges = np.histogram(scores, bins=bins)

        # 전체 scores 개수
        total_scores = len(scores)

        # 각 범위별 비율 계산
        percentages = (counts / total_scores) * 100

        # 그래프 그리기
        plt.figure(figsize=(10, 6))

        # 히스토그램
        plt.hist(scores, bins=bins, color='blue', alpha=0.7, label='Scores Distribution')

        # 평균선
        mean_value = np.mean(scores)
        plt.axvline(x=mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

        # 최대값과 최소값 계산
        max_value = np.max(scores)
        min_value = np.min(scores)

        # 최대값 선 추가
        plt.axvline(x=max_value, color='green', linestyle='--', label=f'Max: {max_value:.2f}')

        # 최소값 선 추가
        plt.axvline(x=min_value, color='blue', linestyle='--', label=f'Min: {min_value:.2f}')

        # 구간별 개수와 비율 텍스트 추가
        for i in range(len(counts)):
            plt.text(bin_edges[i] + 0.05, counts[i] + 0.5,
                     f'{counts[i]} ({percentages[i]:.1f}%)',
                     color='black', ha='center', va='bottom',  fontsize=7)

        # 제목, 라벨, 범례 설정
        plt.title(f"Scores Distribution - {dataset_name} (Total: {total_scores} Samples)")
        plt.xlabel("Score Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        # 이미지 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

        LOGGER.info(f"Plot saved at {save_path}")
        # print(f"Plot saved at {save_path}")

    def test(self, training_data, test_data):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        print("Using ckpt:",ckpt_path)

        state_dict = {}
        if os.path.exists(ckpt_path):
            # 체크포인트 파일이 존재하면 해당 파일에서 상태를 불러옴
            state_dict = torch.load(ckpt_path, map_location=self.device)  # 파일을 현재 사용하는 장치(CPU 또는 GPU)에서 불러오기
            if 'discriminator' in state_dict:  # 불러온 상태 딕셔너리에 Discriminator가 포함되어 있으면
                self.discriminator.load_state_dict(state_dict['discriminator'])  # Discriminator의 상태를 불러온 값으로 초기화
                if "pre_projection" in state_dict:  # pre_projection 레이어 상태가 포함된 경우
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])  # pre_projection의 상태를 불러온 값으로 초기화
            else:
                self.load_state_dict(state_dict, strict=False)  # 체크포인트에 Discriminator가 없으면, 전체 모델의 상태를 불러옴

        all_scores_all_datasets = []
        test_timing = {
            "total_images": 0,
            "total_seconds": 0.0,
            "num_datasets": len(test_data),
        }
        for test_dataloader in test_data:
            LOGGER.info(f"Testing on dataset: {test_dataloader.name}")
            all_scores = []
            dataset_start_time = time.perf_counter()
            for batch in test_dataloader:
                # print(f"Batch type: {type(batch)}")

                # 이미지 데이터 가져오기
                if isinstance(batch, dict) and "image" in batch:
                    images = batch["image"]  # "image" 키에서 이미지 데이터 추출
                    # print(f"Images shape: {images.shape}")
                else:
                    LOGGER.error("Batch does not contain 'image'. Skipping this batch.")
                    continue  # 다음 batch로 넘어감

                batch_size = images.shape[0]
                scores, segmentations, features = self.predict(images)

                all_scores.extend(scores)  # 각 배치의 scores를 하나의 리스트로 합침
                test_timing["total_images"] += batch_size

            test_timing["total_seconds"] += time.perf_counter() - dataset_start_time

            test_data_name = getattr(test_dataloader, "name", "test_data")  # test_data에 'name' 속성이 있으면 사용하고, 없으면 기본값 사용

            # scores 데이터를 엑셀로 저장
            excel_path = os.path.join(self.model_dir, f'savefile/excel/{test_data_name}_scores.xlsx')
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)  # 디렉토리 생성
            scores_df = pd.DataFrame({"Scores": all_scores})  # scores를 데이터프레임으로 변환
            scores_df.to_excel(excel_path, index=False)  # 엑셀로 저장
            LOGGER.info(f"Scores saved at {excel_path}")
            all_scores_all_datasets.append(all_scores)

        savefile_path = os.path.join(self.model_dir, "savefile")
        confusion_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "confusion_matrix2.py"
        )
        confusion_env = os.environ.copy()
        confusion_env["DAAD_CONFUSION_BASE_SAVE_PATH"] = savefile_path
        confusion_env["DAAD_TEST_TOTAL_IMAGES"] = str(test_timing["total_images"])
        confusion_env["DAAD_TEST_TOTAL_SECONDS"] = f"{test_timing['total_seconds']:.6f}"
        confusion_env["DAAD_TEST_NUM_DATASETS"] = str(test_timing["num_datasets"])
        subprocess.run([sys.executable, confusion_script_path], check=True, env=confusion_env)

        # evaluation_results.txt만 run 폴더에 남기고 중간 산출물은 삭제한다.
        shutil.rmtree(self.model_dir, ignore_errors=True)
        models_parent = os.path.dirname(self.model_dir)
        if os.path.isdir(models_parent) and not os.listdir(models_parent):
            os.rmdir(models_parent)

        return all_scores_all_datasets

    def train(self, training_data, fake_data,  test_data):

        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")  # 모델 체크포인트 경로 설정
        if os.path.exists(ckpt_path):
            # 체크포인트 파일이 존재하면 해당 파일에서 상태를 불러옴
            state_dict = torch.load(ckpt_path, map_location=self.device) # 파일을 현재 사용하는 장치(CPU 또는 GPU)에서 불러오기
            if 'discriminator' in state_dict: # 불러온 상태 딕셔너리에 Discriminator가 포함되어 있으면
                self.discriminator.load_state_dict(state_dict['discriminator']) # Discriminator의 상태를 불러온 값으로 초기화
                if "pre_projection" in state_dict: # pre_projection 레이어 상태가 포함된 경우
                    self.pre_projection.load_state_dict(state_dict["pre_projection"]) # pre_projection의 상태를 불러온 값으로 초기화
            else:
                self.load_state_dict(state_dict, strict=False) # 체크포인트에 Discriminator가 없으면, 전체 모델의 상태를 불러옴

        # self.predict(training_data, "train_")  # training_data를 사용하여 학습 데이터의 예측을 수행

        def update_state_dict():
            """현재 Discriminator와 Pre-Projection의 상태를 저장"""
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu() for k, v in self.discriminator.state_dict().items()
            })
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu() for k, v in self.pre_projection.state_dict().items()
                })
            return state_dict

        # Early Stopping 변수
        best_loss = float("inf")
        patience = 3
        counter = 0
        best_state_dict = None
        min_delta = 1e-5

        # Loss 값을 저장할 리스트
        loss_history = []

        start_time = time.time()

        for i_mepoch in range(self.meta_epochs):
            epoch_start_time = time.time()
            print(f"Epoch {i_mepoch + 1}/{self.meta_epochs}")

            # Discriminator 학습 수행
            all_loss = self._train_discriminator(training_data, fake_data)
            delta_loss = best_loss - all_loss

            # Loss 기록
            loss_history.append(all_loss)

            # Early Stopping 체크
            if delta_loss > min_delta:
                best_loss = all_loss
                counter = 0
                best_state_dict = update_state_dict()
                torch.save(best_state_dict, ckpt_path)
                print(f"New best loss: {best_loss:.5f} at epoch {i_mepoch + 1}")
            else:
                counter += 1
                print(f"Early stopping counter: {counter}/{patience}")

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"🔹 **Epoch {i_mepoch + 1} Time Taken: {epoch_duration:.2f} seconds**")

            if counter >= patience:
                print("Early stopping triggered. Restoring best model...")
                if best_state_dict is not None:  # None 체크 추가
                    torch.save(best_state_dict, ckpt_path)
                    del best_state_dict
                    torch.cuda.empty_cache()
                break

            gc.collect()
            torch.cuda.empty_cache()


        total_duration = time.time() - start_time

        hours = total_duration // 3600
        minutes = (total_duration % 3600) // 60
        seconds = total_duration % 60

        # 학습 시간 출력
        print(f"🔹 **Total Training Time: {int(hours):02}:{int(minutes):02}:{int(seconds):02} (HH:MM:SS)**")

        # 1. 일반 스케일로 Loss Plot 저장
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs (Linear Scale)')
        plt.legend()
        plt.grid(True)

        # 그래프 여백을 자동으로 조정
        plt.tight_layout()

        loss_path = os.path.join(self.model_dir, 'loss')
        os.makedirs(loss_path, exist_ok=True)
        # Loss plot을 이미지 파일로 저장 (일반 스케일)
        linear_loss_plot_path = os.path.join(loss_path, 'loss_plot_linear.png')
        plt.savefig(linear_loss_plot_path)
        print(f"Linear Loss plot saved to {linear_loss_plot_path}")
        plt.clf()
        plt.close()

        # 2. 로그 스케일로 Loss Plot 저장
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs (Log Scale)')
        plt.legend()
        plt.grid(True)

        # y축을 로그 스케일로 설정
        plt.yscale('log')
        # 그래프 여백을 자동으로 조정
        plt.tight_layout()

        # Loss plot을 이미지 파일로 저장 (로그 스케일)
        log_loss_plot_path = os.path.join(loss_path, 'loss_plot_log.png')
        plt.savefig(log_loss_plot_path)
        print(f"Log Loss plot saved to {log_loss_plot_path}")
        plt.clf()
        plt.close()

        # **loss_history 저장**: pickle을 사용하여 loss_history를 동일한 경로에 저장
        loss_history_path = os.path.join(loss_path, 'loss_history.pkl')
        with open(loss_history_path, 'wb') as f:  # **pickle로 저장**
            pickle.dump(loss_history, f)
        print(f"Loss history saved to {loss_history_path}")

    def _train_discriminator(self, input_data, fake_input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval() # 모든 forward 모듈을 평가 모드로 설정
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator = self.discriminator.to(self.device)
        self.discriminator.train()

        LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar: # GAN 훈련 에폭에 대한 진행률 표시줄 설정
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                # all_p_interp = []
                # embeddings_list = []

    ########################################################################################################################################



########################################################################################################################################
                # Feature Level Augmentation / Feature selection


                for data_item, fake_data_item in itertools.zip_longest(input_data, fake_input_data, fillvalue=None): # 길이가 긴 쪽 기준으로 채움

                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()

                    true_feats = None
                    fake_feats = None
########################################################################################################################################
                    # real
                    if data_item is not None:
                        # 입력 이미지와 그레이디언트 정보 이동
                        img = data_item["image"]
                        img = img.to(torch.float).to(self.device)
                        # 사전 프로젝션이 있으면 임베딩된 특징을 추출, 없으면 바로 특징을 추출
                        if self.pre_proj > 0:
                            true_feats, patch_shapes = self._embed(img, evaluation=False)  # features와 patch_shapes를 모두 받음
                            patch_gap = patch_shapes[0][0] * patch_shapes[0][1] # 한 이미지에서 나오는 패치 개수
                            true_feats = self.pre_projection(true_feats)  # pre_projection 적용
                        else:
                            true_feats = self._embed(img, evaluation=False)[0]

                        # print(f"[REAL] FLA 전 feature shape: {true_feats.shape}")
                        true_feats = self.apply_feature_augmentation(feats=true_feats, case=int(self.true_FLA), patch_gap=patch_gap,image_gap=self.image_gap)
                        # print(f"[REAL] FLA 후 feature shape: {true_feats.shape}")

                        # true feature에 대해 FS 적용
                        true_feats = self.apply_feature_selection(feats=true_feats,keep_ratio=self.fs_keep_ratio,mode=self.fs_mode)
                        # print(f"[REAL] FS 후 feature shape: {true_feats.shape}")
########################################################################################################################################
                    # fake
                    if fake_data_item is not None:
                        fake_img = fake_data_item["image"]
                        fake_img = fake_img.to(torch.float).to(self.device)

                        if self.pre_proj > 0:
                            false_feats, patch_shapes = self._embed(fake_img, evaluation=False)
                            patch_gap = patch_shapes[0][0] * patch_shapes[0][1]
                            false_feats = self.pre_projection(false_feats)
                        else:
                            false_feats = self._embed(fake_img, evaluation=False)[0]

                        # print(f"[FAKE] FLA 전 feature shape: {false_feats.shape}")
                        fake_feats = self.apply_feature_augmentation(feats=false_feats, case=int(self.false_FLA), patch_gap=patch_gap,image_gap=self.image_gap)
                        # print(f"[FAKE] FLA 후 feature shape: {fake_feats.shape}")
                        # fake feature에 대해 FS 적용
                        fake_feats = self.apply_feature_selection(feats=fake_feats,keep_ratio=self.fs_keep_ratio,mode=self.fs_mode)
                        # print(f"[FAKE] FS 후 feature shape: {fake_feats.shape}")
########################################################################################################################################
                    # 하나라도 None인 경우 빈 텐서로 대체
                    if true_feats is None and fake_feats is not None: # true_feats = None이면 fake_feats의 채널 수(C)를 기준으로 [0, C] 텐서를 만듦
                        true_feats = torch.zeros((0, fake_feats.shape[1]), device=self.device) # 빈 텐서 추가
                    elif true_feats is None:
                        continue  # 둘 다 None이면 스킵

                    if fake_feats is None and true_feats is not None:
                        fake_feats = torch.zeros((0, true_feats.shape[1]), device=self.device) # 빈 텐서 추가
                    elif fake_feats is None:
                        continue  # 둘 다 None이면 스킵


                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                    true_scores = scores[:len(true_feats)] if len(true_feats) > 0 else torch.tensor([], device=self.device)
                    fake_scores = scores[len(true_feats):] if len(fake_feats) > 0 else torch.tensor([], device=self.device)


                    th = self.dsc_margin
                    # 참과 가짜 샘플에 대한 확률 계산
                    # p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    # p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    # true_loss = torch.clip(-true_scores + th, min=0)
                    # fake_loss = torch.clip(fake_scores + th, min=0)

                    if len(true_scores) > 0:
                        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                        # true_loss = torch.clip(-true_scores + th, min=0)
                        true_loss = torch.clip(true_scores + th, min=0)
                    else:
                        p_true = torch.tensor(0.0, device=self.device)  # 기본값 설정
                        true_loss = torch.tensor(0.0, device=self.device)  # 빈 텐서 대신 0 텐서

                    if len(fake_scores) > 0:
                        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                        # fake_loss = torch.clip(fake_scores + th, min=0)
                        fake_loss = torch.clip(-fake_scores + th, min=0)
                    else:
                        p_fake = torch.tensor(0.0, device=self.device)  # 기본값 설정
                        fake_loss = torch.tensor(0.0, device=self.device)  # 빈 텐서 대신 0 텐서





                    # 확률과 손실 값을 로그에 기록
                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)

                    loss = true_loss.mean() + fake_loss.mean()
                    if not torch.isnan(loss) and loss > 0:
                        self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                        self.logger.step()
                        loss.backward()

                        # 옵티마이저 업데이트
                        if self.pre_proj > 0:
                            self.proj_opt.step()
                        if self.train_backbone:
                            self.backbone_opt.step()
                        self.dsc_opt.step()

                        all_loss.append(loss.detach().cpu().item())
                        all_p_true.append(p_true.detach().cpu().item())
                        all_p_fake.append(p_fake.detach().cpu().item())
                        # all_p_true.append(p_true.cpu().item())
                        # all_p_fake.append(p_fake.cpu().item())

                        del loss


                    torch.cuda.empty_cache()
                    gc.collect()
    ########################################################################################################################################

                # if len(embeddings_list) > 0:
                #     self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)

                if self.cos_lr:
                    self.dsc_schl.step()

                num_real = len(input_data)
                num_fake = len(fake_input_data)
                denom = num_real + num_fake

                all_loss = sum(all_loss) / denom
                all_p_true = sum(all_p_true) / num_real
                all_p_fake = sum(all_p_fake) / num_fake
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                # 진행 상태를 포함한 진행률 표시줄 업데이트
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                # if len(all_p_interp) > 0:
                #     pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)
        return all_loss


    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()


        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, features, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        # print(f"Type of images: {type(images)}")
        # images가 문자열인 경우 텐서로 변환
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():  # 추론을 할 때는 gradient 계산을 하지 않도록 with torch.no_grad() 사용
            features, patch_shapes = self._embed(images,# 이미지를 임베딩하고 패치 크기를 제공 (특징 추출 및 패치 정보 얻기)
                                                 provide_patch_shapes=True, 
                                                 evaluation=True)
            if self.pre_proj > 0:# 만약 사전 프로젝션이 있다면 적용
                features = self.pre_projection(features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            # patch_scores = image_scores = -self.discriminator(features) # Discriminator를 통해 얻은 features에 대해 점수 계산
            patch_scores = image_scores = self.discriminator(features)   # 원래 dis는 정상을 크게 비정상을 작게 나오도록 설정하고 나중에 score구할때 -붙여서 계산하는데 이거 수정
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores( # 이미지 점수를 unpatching (원래 크기로 되돌리기)
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1) # 점수를 적절히 재구성 (차원 맞추기)
            image_scores = self.patch_maker.score(image_scores) # patch별 점수 계산

            patch_scores = self.patch_maker.unpatch_scores( # 패치 점수도 unpatching (원래 크기로 되돌리기)
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]  # 패치의 크기 정보 얻기
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1]) # 패치 점수 형태를 배치 크기, 패치 크기에 맞게 재구성
            features = features.reshape(batchsize, scales[0], scales[1], -1)  # features를 패치 크기에 맞게 재구성
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features) # anomaly segmentation (이상 탐지 마스크 생성)

        return list(image_scores), list(masks), list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

# Image handling classes.
class PatchMaker:
    # 클래스 초기화: 패치 크기, top_k (최상위 k 값), stride(스트라이드) 설정
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches. 주어진 텐서를 패치 텐서로 변환
        Args:
            x: [torch.Tensor, bs x c x w x h] (배치 크기 x 채널 x 너비 x 높이)
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2) # 패치 크기에 맞춰 패딩 크기 계산
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features) # 텐서를 펼쳐서 패치 형태로 변환
        # 이미지 크기와 패치 크기, 스트라이드를 기준으로 전체 패치 개수 계산
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        # 텐서를 패치 크기 (patchsize x patchsize) 형태로 리쉐이핑
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)  # 차원 순서 변경

        if return_spatial_info:
            return unfolded_features, number_of_total_patches  # 패치와 총 패치 개수를 반환
        return unfolded_features # 패치만 반환

    def unpatch_scores(self, x, batchsize):
        # 패치된 점수들을 배치 크기만큼 다시 원래 형태로 되돌림
        return x.reshape(batchsize, -1, *x.shape[1:]) # 배치 크기에 맞춰 리쉐이프

    def score(self, x): # 텐서의 점수를 계산하는 함수 (top_k값에 따라 처리)
        was_numpy = False
        if isinstance(x, np.ndarray):   # 입력이 numpy 배열이면 torch 텐서로 변환
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:  # 텐서의 차원이 2보다 크면 차원 축소
            x = torch.max(x, dim=-1).values  # 마지막 차원에서 최대값 추출
        if x.ndim == 2:  # 2D 텐서일 경우
            if self.top_k > 1:   # top_k가 1보다 크면 top-k 값 계산
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)  # top-k 평균값 계산
            else:  # 최대값 계산
                x = torch.max(x, dim=1).values
        if was_numpy: # 원래 numpy 배열이었다면 다시 numpy로 변환
            return x.numpy()
        return x


# Compatibility alias for project-facing DAAD naming.
DAAD = SimpleNet
