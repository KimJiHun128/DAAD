# ------------------------------------------------------------------
# DAAD: CTCA anomaly detection experiments built on the SimpleNet codebase.
# Original SimpleNet source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------
# 명령어로 인자값 받아서 실행시키는 코드

import logging
import os
import sys
from torchvision import transforms
import click
import numpy as np
import torch

sys.path.append("src")
import backbones
import common
import metrics
import simplenet 
import utils
import torchvision.utils as vutils
import random
import time
LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "CTCA": ["datasets.mvtec", "MVTecDataset"],
}


# 각 옵션은 CLI 명령어에 옵션으로 전달되어, 그 값을 kwargs를 통해 main() 함수로 전달
@click.group(chain=True) # @click.group()은 여러 명령어를 하나의 그룹으로 묶을 때 사용
@click.option("--results_path", type=str, default="/home/jihun/PycharmProjects/DAAD/results")
@click.option("--gpu", type=int, default=[0,1], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str,default="daad_ctca")
@click.option("--log_project", type=str, default="CTCA")
@click.option("--run_name", type=str, default="run")
@click.option("--test", is_flag=True,  default=False)
@click.option("--save_segmentation_images", is_flag=True, default=True, show_default=True)
def main(**kwargs):
    pass


@main.result_callback() # main 완료된 후에 추가적인 작업을 수행
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test,
    save_segmentation_images
):
    # seed = int(time.time())
    methods = {key: item for (key, item) in methods} # methods 딕셔너리의 키-값을 다시 딕셔너리로 구성

    net_config = getattr(methods["get_simplenet"], "folder_config", {})
    dataset_config = getattr(methods["get_dataloaders"], "folder_config", {})
    base_run_name = run_name
    if base_run_name == "run":
        base_run_name = ""
    elif base_run_name.startswith("run_"):
        base_run_name = base_run_name[len("run_"):]

    condition_parts = []
    if dataset_config:
        if "cv_fold" in dataset_config and "cv_num_folds" in dataset_config:
            condition_parts.append(f"CVfold{dataset_config['cv_fold']}of{dataset_config['cv_num_folds']}")
        if dataset_config["real_aug"] == dataset_config["fake_aug"]:
            condition_parts.append(f"ILA{dataset_config['real_aug']}")
        else:
            condition_parts.extend([
                f"realaug{dataset_config['real_aug']}",
                f"fakeaug{dataset_config['fake_aug']}",
            ])
    if net_config:
        condition_parts.extend([
            f"truefla{net_config['true_fla']}",
            f"fakefla{net_config['false_fla']}",
            f"FS{net_config['fs_mode']}",
        ])
    condition_name = "_".join(condition_parts)
    if base_run_name and condition_name:
        run_name = f"{base_run_name}_{condition_name}"
    elif condition_name:
        run_name = condition_name
    else:
        run_name = base_run_name
    LOGGER.info(f"Run folder name: {run_name}")

    # 결과를 저장할 폴더를 생성. 이미 존재하면 덮어씀.
    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )
    # 현재 프로세스 ID를 가져옴
    pid = os.getpid()
    # 데이터 로더를 가져오는 함수 실행 (방법으로 제공된 methods["get_dataloaders"])
    list_of_dataloaders = methods["get_dataloaders"](seed)
    # GPU 설정
    device = utils.set_torch_device(gpu)

    result_collect = []
    # 데이터 로더 목록을 순차적으로 평가
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders): # only CTCA
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name# 현재 데이터셋 이름

        imagesize = dataloaders["training"].dataset.imagesize# 데이터셋의 이미지 크기
        simplenet_list = methods["get_simplenet"](imagesize, device)# methods 딕셔너리에서 DAAD 모델을 반환
        # 모델을 저장할 폴더 경로
        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)

        save_aug_image = 0 # 1 save aug image

        if save_aug_image == 1:
            num_batches_to_save = 5  # 저장할 배치 수 (조정 가능)
            images_per_batch = 16  # 한 배치에서 저장할 이미지 수

            images_dir = os.path.join(models_dir, "aug_images")

                # Augment된 이미지 저장 폴더 설정
            augmented_save_dir = os.path.join(images_dir, "augmented_train_images")
            original_save_dir = os.path.join(images_dir, "original_augment_train_images")
            fake_augmented_save_dir = os.path.join(images_dir, "fake_augmented_train_images")  # Fake augmented 이미지 저장 폴더
            fake_original_save_dir = os.path.join(images_dir,"fake_original_augment_train_images")  # Fake original 이미지 저장 폴더
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(augmented_save_dir, exist_ok=True)
            os.makedirs(original_save_dir, exist_ok=True)
            os.makedirs(fake_augmented_save_dir, exist_ok=True)  # Fake augmented 폴더 생성
            os.makedirs(fake_original_save_dir, exist_ok=True)  # Fake original 폴더 생성


            # 이미지 크기 변경을 위한 Transform 정의
            resize_transform = transforms.Resize((128, 128))  # 원하는 크기로 설정

            # Train 데이터셋에서 이미지 저장
            train_dataloader = dataloaders["training"]
            fake_dataloader = dataloaders["fake"]  # Fake 데이터 로더


            for batch_idx, batch in enumerate(train_dataloader):
                images = batch["image"]  # 배치에서 이미지 가져오기

                # 원본 이미지 저장 (보간 X)
                original_save_path = os.path.join(original_save_dir, f"train_batch{batch_idx}.png")
                vutils.save_image(images[:images_per_batch], original_save_path, nrow=4, normalize=True)
                print(f"Saved {images_per_batch} original images from batch {batch_idx} to {original_save_path}")

                # 이미지 크기 변경 후 저장 (보간 O)
                resized_images = torch.stack([resize_transform(image) for image in images])
                resized_save_path = os.path.join(augmented_save_dir, f"train_batch{batch_idx}.png")
                vutils.save_image(resized_images[:images_per_batch], resized_save_path, nrow=4, normalize=True)
                print(f"Saved {images_per_batch} resized images from batch {batch_idx} to {resized_save_path}")

                # 지정한 배치 수만큼 저장하고 종료
                if batch_idx + 1 >= num_batches_to_save:
                    break

            # Fake 데이터셋에서 원본 및 augment된 이미지 저장
            for batch_idx, batch in enumerate(fake_dataloader):
                fake_images = batch["image"]  # Fake 배치에서 이미지 가져오기

                # Fake 원본 이미지 저장 (보간 X)
                fake_original_save_path = os.path.join(fake_original_save_dir, f"fake_batch{batch_idx}.png")
                vutils.save_image(fake_images[:images_per_batch], fake_original_save_path, nrow=4, normalize=True)
                print(f"Saved {images_per_batch} fake original images from batch {batch_idx} to {fake_original_save_path}")

                # Fake 이미지 크기 변경 후 저장 (보간 O)
                resized_fake_images = torch.stack([resize_transform(image) for image in fake_images])
                fake_resized_save_path = os.path.join(fake_augmented_save_dir, f"fake_batch{batch_idx}.png")
                vutils.save_image(resized_fake_images[:images_per_batch], fake_resized_save_path, nrow=4, normalize=True)
                print(f"Saved {images_per_batch} fake resized images from batch {batch_idx} to {fake_resized_save_path}")

                # 지정한 배치 수만큼 저장하고 종료
                if batch_idx + 1 >= num_batches_to_save:
                    break

        # 모델 리스트를 순차적으로 학습
        for i, daad_model in enumerate(simplenet_list):
            torch.cuda.empty_cache()
            # DAAD 백본 모델에 시드 설정 (필요한 경우)
            if daad_model.backbone.seed is not None:
                utils.fix_seeds(daad_model.backbone.seed, device)
            LOGGER.info(
                "Training models ({}/{})".format(i + 1, len(simplenet_list))# 몇 번째 모델을 학습 중인지 출력
            )
            # torch.cuda.empty_cache()
            # 모델을 저장할 디렉토리 설정
            daad_model.set_model_dir(os.path.join(models_dir, f"{i}"), dataset_name)
            # 테스트 모드가 아니면 학습을 진행, 그렇지 않으면 테스트 진행
            if not test:
                daad_model.train(dataloaders["training"], dataloaders["fake"], dataloaders["testing"])
                daad_model.test(dataloaders["training"], dataloaders["testing"])
                # score = daad_model.test(dataloaders["training"], dataloaders["testing"])
            else:
                print("Start testing mode")
                daad_model.test(dataloaders["training"], dataloaders["testing"])
                # score = daad_model.test(dataloaders["training"], dataloaders["testing"])


            # result_collect.append(
            #     {
            #         "dataset_name": dataset_name,
            #         # "instance_auroc": i_auroc, # auroc,
            #         # "full_pixel_auroc": p_auroc, # full_pixel_auroc,
            #         # "anomaly_pixel_auroc": pro_auroc, # 이상 픽셀 AUROC
            #         "score" : score
            #     }
            # )
            # # 결과 출력
            # for key, item in result_collect[-1].items():
            #     if key != "dataset_name":# 'dataset_name' 제외하고 출력
            #         LOGGER.info("{0}: {1:3.3f}".format(key, item))

        LOGGER.info("\n\n-----\n")
    # Store all results and mean scores to a csv-file. # 모든 결과를 저장하고 평균 성능을 CSV 파일에 저장
    # result_metric_names = list(result_collect[-1].keys())[1:]
    # result_dataset_names = [results["dataset_name"] for results in result_collect]
    # result_scores = [list(results.values())[1:] for results in result_collect]
    # 결과를 계산하고 CSV 파일에 저장하는 함수 호출
    # utils.compute_and_store_final_results(
    #     run_save_path,
    #     result_scores,
    #     column_names=result_metric_names,
    #     row_names=result_dataset_names,
    # )


@main.command("net")  # CLI 명령어 'net'을 정의합니다.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=["resnet18"])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True) # backbone에서 추출할 layer들 리스트
@click.option("--pretrain_embed_dimension", type=int, default=256)
@click.option("--target_embed_dimension", type=int, default=256)
@click.option("--patchsize", type=int, default=3)
@click.option("--embedding_size", type=int, default=64)
@click.option("--meta_epochs", type=int, default=10)
@click.option("--aed_meta_epochs", type=int, default=1)
@click.option("--gan_epochs", type=int, default=4)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=256)
@click.option("--noise_std", type=float, default=0.015)
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--dsc_lr", type=float, default=0.0002)
@click.option("--auto_noise", type=float, default=0)
@click.option("--train_backbone", is_flag=True)
@click.option("--cos_lr", is_flag=True)
@click.option("--pre_proj", type=int, default=1)
@click.option("--proj_layer_type", type=int, default=0)
@click.option("--mix_noise", type=int, default=1)
@click.option("--patchstride", type=int, default=6)  # patchstride 추가
@click.option("--false_fla", type=click.Choice(["0", "1", "2", "3", "5", "6"]), default="1")
@click.option("--true_fla", type=click.Choice(["0", "1", "2", "3", "5", "6"]), default="1")
@click.option("--fs_keep_ratio", type=float, default=0.5, help="Feature selection 남길 비율")
@click.option("--fs_mode", type=click.Choice(["gap", "gmp", "both", "none"], case_sensitive=False), default="both", help="Feature selection 모드")
@click.option("--image_gap", type=int, default=5, help="Image gap 값")
def net(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    patchstride,
    embedding_size,
    meta_epochs,
    aed_meta_epochs,
    gan_epochs,
    noise_std,
    dsc_layers, 
    dsc_hidden,
    dsc_margin,
    dsc_lr,
    auto_noise,
    train_backbone,
    cos_lr,
    pre_proj,
    proj_layer_type,
    mix_noise,
    false_fla,
    true_fla,
    fs_keep_ratio,
    fs_mode,
    image_gap,
):
    fs_mode = fs_mode.lower()
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1: # backbone_names가 여러 개일 경우
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))] # 각 backbone에 대한 layer들을 담을 리스트
        for layer in layers_to_extract_from:  # layers_to_extract_from을 각 backbone에 맞게 나눔
            idx = int(layer.split(".")[0])  # "backbone_name.layer_name" 형식에서 backbone을 구분
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer) # 각 backbone에 해당하는 layer들을 분배
    else: # backbone_names가 하나일 경우
        layers_to_extract_from_coll = [layers_to_extract_from] # layers_to_extract_from을 그대로 사용

    def get_simplenet(input_shape, device):
        simplenets = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ): # 각 backbone과 해당하는 layer들을 순차적으로 처리
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)  # backbone 모델 로드
            backbone.name, backbone.seed = backbone_name, backbone_seed  # backbone 이름과 seed 값 설정

            simplenet_inst = simplenet.SimpleNet(device) # DAAD 모델 인스턴스 생성
            simplenet_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                patchstride=patchstride,
                embedding_size=embedding_size,
                meta_epochs=meta_epochs,
                aed_meta_epochs=aed_meta_epochs,
                gan_epochs=gan_epochs,
                noise_std=noise_std,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
                auto_noise=auto_noise,
                train_backbone=train_backbone,
                cos_lr=cos_lr,
                pre_proj=pre_proj,
                proj_layer_type=proj_layer_type,
                mix_noise=mix_noise,
                false_FLA=false_fla,
                true_FLA=true_fla,
                fs_keep_ratio=fs_keep_ratio,
                fs_mode=fs_mode,
                image_gap=image_gap,
            )
            simplenets.append(simplenet_inst)
        return simplenets

    false_fla = int(false_fla)
    true_fla = int(true_fla)

    def fla_to_str(fla_code):
        if fla_code == 0:
            return "원본 그대로 사용"
        elif fla_code == 1:
            return "원본 + 노이즈"
        elif fla_code == 2:
            return "원본 + 내삽"
        elif fla_code == 3:
            return "원본 + 외삽"
        elif fla_code == 5:
            return "내삽 + 외삽"
        elif fla_code == 6:
            return "noise + 내삽 + 외삽 "
        else:
            return "알 수 없는 FLA 코드"

    print(f"[Fake FLA 설정] {false_fla}: {fla_to_str(false_fla)}")
    print(f"[Real FLA 설정] {true_fla}: {fla_to_str(true_fla)}")
    print(f"[Image Gap 설정] image_gap: {image_gap}")
    # Feature Selection 설정 출력
    print(f"[FS 설정] mode: {fs_mode}, keep_ratio: {fs_keep_ratio}, ")

    get_simplenet.folder_config = {
        "true_fla": true_fla,
        "false_fla": false_fla,
        "fs_mode": fs_mode,
    }

    return ("get_simplenet", get_simplenet) # 함수 이름과 함수를 반환


@main.command("dataset")
@click.argument("name", type=str, default="CTCA") # 데이터셋 이름을 인자로 받음
@click.argument("data_path", type=click.Path(exists=True, file_okay=False), default="/home/jihun/PycharmProjects/DAAD/CTCA_8")
@click.option("--subdatasets", "-d", multiple=True, type=str)# 하위 데이터셋 이름을 여러 개 받을 수 있음
@click.option("--train_val_split", type=float, default=1, show_default=True)  # 학습/검증 데이터 비율 (디폴트 1)
@click.option("--batch_size", default=1024, type=int, show_default=True)
@click.option("--num_workers", default=2, type=int, show_default=True)
@click.option("--resize", default=64, type=int, show_default=True)
@click.option("--imagesize", default=64, type=int, show_default=True)

@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0,type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)

@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)

@click.option("--fake_aug", default=0, type=int, show_default=True) #1로 설정 시 fake augmentation 사용
@click.option("--real_aug", default=0, type=int, show_default=True) #1로 설정 시 real augmentation 사용
@click.option("--cv_fold", default=1, type=int, show_default=True, help="Normal-only cross-validation fold index. 1-based.")
@click.option("--cv_num_folds", default=4, type=int, show_default=True, help="Number of normal-only cross-validation folds.")
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    rotate_degrees,
    translate,
    scale,
    brightness,
    contrast,
    saturation,
    gray,
    hflip,
    vflip,
    real_aug,
    fake_aug,
    cv_fold,
    cv_num_folds,
):
    dataset_info = _DATASETS[name] # dataset_info는 'name'에 해당하는 데이터셋에 대한 정보 딕셔너리 "mvtec"
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]]) # dataset_library는 데이터셋을 로드하는 라이브러리

    def get_dataloaders(seed):
        dataloaders = []  # 생성될 DataLoader 객체들을 담을 리스트
        normal_subdatasets = [f"N_{idx}" for idx in range(1, 21)]
        disease_subdatasets = [f"D_{idx}" for idx in range(1, 21)]

        if cv_num_folds <= 0:
            raise click.ClickException("--cv_num_folds must be greater than 0.")
        if cv_fold < 1 or cv_fold > cv_num_folds:
            raise click.ClickException(f"--cv_fold must be between 1 and {cv_num_folds}.")
        if len(normal_subdatasets) % cv_num_folds != 0:
            raise click.ClickException("The number of normal subdatasets must be divisible by --cv_num_folds.")

        fold_size = len(normal_subdatasets) // cv_num_folds
        test_start = (cv_fold - 1) * fold_size
        test_end = test_start + fold_size
        test_normals = normal_subdatasets[test_start:test_end]
        train_normals = normal_subdatasets[:test_start] + normal_subdatasets[test_end:]
        subdatasets = disease_subdatasets + test_normals

        train_classnames = [f"test/{normal_name}" for normal_name in train_normals]
        LOGGER.info(
            f"CV fold {cv_fold}/{cv_num_folds}: train_normal={train_normals}, "
            f"test_normal={test_normals}, test_disease={disease_subdatasets}"
        )
        # train_dataset 생성: 학습용 데이터셋 생성
        train_dataset = dataset_library.__dict__[dataset_info[1]]( # datasets.mvtec MVTecDataset 클래스의 인스턴스를 생성
            data_path,
            # classname='png/Train/train/good',
            classname=train_classnames,
            # classname='train_real_10_30',
            resize=resize,
            train_val_split=train_val_split,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=seed,
            rotate_degrees=rotate_degrees,
            translate=translate,
            brightness_factor=brightness,
            contrast_factor=contrast,
            saturation_factor=saturation,
            gray_p=gray,
            h_flip_p=hflip,
            v_flip_p=vflip,
            scale=scale,
            real_aug=real_aug,
            fake_aug=0,
        )
        # DataLoader 생성: 학습용 DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # prefetch_factor=2,
            pin_memory=True,
        )
        LOGGER.info(f"Dataset: train_real={len(train_dataset)} ")
        train_dataloader.name = name

        val_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=train_classnames,
            # classname='train_fake_10_30',
            resize=resize,
            train_val_split=train_val_split,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=seed,
            fake_aug = fake_aug,
            real_aug=0,
        )
        fake_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # prefetch_factor=2,
            pin_memory=True,
        )

        LOGGER.info(f"Dataset: train_fake={len(val_dataset)} ")

        # if subdataset is not None:
        #     train_dataloader.name += "_" + subdataset


        # 하나의 테스트 데이터셋을 여러 서브데이터셋에 대해 테스트하도록 설정
        test_dataloaders = []  # 테스트용 dataloader들을 저장할 리스트

        for subdataset in subdatasets: # 여러 subdataset에 대해 처리

            # test_dataset 생성: 테스트용 데이터셋 생성
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname='test/'+subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            # 테스트용 DataLoader
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                # prefetch_factor=2,
                pin_memory=True,
            )
            test_dataloader.name = subdataset  # 서브데이터셋 이름 설정
            test_dataloaders.append(test_dataloader)  # test_dataloader 리스트에 추가

            LOGGER.info(f"Dataset: test= {subdataset} {len(test_dataset)}")


        # # 검증 데이터셋 생성: train_val_split이 1 미만일 때만 생성
        # if train_val_split < 1:
        #     val_dataset = dataset_library.__dict__[dataset_info[1]](
        #         data_path,
        #         classname=subdataset,
        #         resize=resize,
        #         train_val_split=train_val_split,
        #         imagesize=imagesize,
        #         split=dataset_library.DatasetSplit.VAL,
        #         seed=seed,
        #     )
        #
        #     val_dataloader = torch.utils.data.DataLoader(
        #         val_dataset,
        #         batch_size=batch_size,
        #         shuffle=False,
        #         num_workers=num_workers,
        #         prefetch_factor=4,
        #         pin_memory=True,
        #     )
        # else:
        #     val_dataloader = None # 검증 데이터셋을 만들지 않음
        dataloader_dict = {
            "training": train_dataloader,
            "fake": fake_dataloader,
            "testing": test_dataloaders,
        }



        dataloaders.append(dataloader_dict)  # 생성된 dataloader들을 리스트에 추가
        return dataloaders # 모든 dataloaders 리스트 반환

    get_dataloaders.folder_config = {
        "real_aug": real_aug,
        "fake_aug": fake_aug,
        "cv_fold": cv_fold,
        "cv_num_folds": cv_num_folds,
    }

    return ("get_dataloaders", get_dataloaders) # 함수 이름과 함수를 반환


if __name__ == "__main__": # 이 파일이 직접 실행될 때
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))  # 실행된 명령어 출력
    main() # main() 함수를 실행하여 CLI 명령어를 실행
