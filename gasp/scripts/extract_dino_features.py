import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tyro
from tqdm import tqdm

from gasp.config.method_configs import convert_markup_to_ansi
from gasp.utils.denoising_dino_feature_extractor import _forward, _get_model, _get_prep


def setup():
    torch.set_float32_matmul_precision("medium")  # heighest, high, medium


ALL_ARGOVERSE_CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)


@dataclass
class ExtractDinoFeaturesConfig:
    # Data i/o
    output_root: Path = Path("/outputs")
    """The output directory."""
    argoverse_input_dir: Path = Path("/datasets/av2")
    """The argoverse dataset directory"""

    dataset: str = "argoverse"
    """The dataset to use. 'argoverse' """
    use_cuda: bool = True
    """ Whether to use cuda """
    pca_use_cuda: bool = True
    """ Whether to use cuda for PCA """
    use_if_exists: bool = True
    """ Whether to force overwrite stuff or use if exists..."""

    # DINO things
    denoiser: Path = Path("/outputs/distilled_imagenet_vit_base_patch14_dinov2.lvd142m.pth")
    """The path to the denoiser model weights"""
    backbone_type: str = "DeViT"
    """ The backbone type """
    model_type: str = "base"
    """ The model type """
    model_stride: int = 7  # 7 for DINOv2, 8 for DINO
    """ The model stride """
    patch_size: int = 14
    """ The patch size """
    enable_pe: bool = True
    """ Whether to enable positional encoding """
    pos_h: int = 37
    pos_w: int = 37
    dino_feature_dim: int = 768
    batch_size: int = 8

    num_subset_for_reduction_matrix: int = 5_000
    """The number of subset files to use for the PCA reduction matrix"""
    num_elements_for_reduction_matrix: int = 500_000

    target_feature_dim: int = 16
    """ if not null, we will use pca to reduce the dimension of 2D features to this value """

    feature_dims_to_rgb: Tuple[int, ...] = (8, 16)
    """ The first number of feature dimensions to convert to RGB """

    # Argoverse specific
    # center image fullres: 2048x1550 [h,w] -> dinores: 1540x1540 [h,w] (to be factor of 14)
    # other images fullres: 1550x2048 [h,w] -> dinores: 1540x2044 [h,w] (to be factor of 14)
    argoverse_cameras_to_use: Tuple = (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    )
    """The cameras to use"""
    argoverse_center_crop: Tuple[int, int, int, int] = (
        286,
        5,
        212,
        5,
    )  # (top, right, bottom, left)
    argoverse_center_size: Tuple[int, int] = (770, 770)  # [h, w]

    argoverse_other_crop: Tuple[int, int, int, int] = (
        2,
        5,
        2,
        5,
    )  # (top, right, bottom, left)
    argoverse_other_size: Tuple[int, int] = (770, 1022)  # [h, w]


def get_shape(config: ExtractDinoFeaturesConfig):
    if config.dataset == "argoverse":
        shape = f"{config.argoverse_other_size[0]}x{config.argoverse_other_size[1]}"
    else:
        raise NotImplementedError("Dataset not implemented")
    return shape


def get_subset_dir(config: ExtractDinoFeaturesConfig):
    shape = get_shape(config)
    subset_dir = config.output_root / f"{config.dataset}_dino_features" / config.model_type / shape / "subset"
    os.makedirs(subset_dir, exist_ok=True)
    return subset_dir


def get_compressed_subset_dir(config: ExtractDinoFeaturesConfig):
    shape = get_shape(config)
    subset_dir = (
        config.output_root / f"{config.dataset}_dino_features" / config.model_type / shape / "compressed_subset"
    )
    os.makedirs(subset_dir, exist_ok=True)
    return subset_dir


def get_compressed_features_dir(config: ExtractDinoFeaturesConfig):
    shape = get_shape(config)
    compressed_features_dir = (
        config.output_root / f"{config.dataset}_dino_features" / config.model_type / shape / "compressed_features"
    )
    os.makedirs(compressed_features_dir, exist_ok=True)
    return compressed_features_dir


def get_pca_reduction_matrix_file(config: ExtractDinoFeaturesConfig):
    shape = get_shape(config)
    pca_reduction_matrix_file = (
        config.output_root / f"{config.dataset}_dino_features" / config.model_type / shape / "pca_reduction_matrix.pt"
    )
    os.makedirs(pca_reduction_matrix_file.parent, exist_ok=True)
    return pca_reduction_matrix_file


def get_reduction_info_file(config: ExtractDinoFeaturesConfig):
    shape = get_shape(config)
    reduction_info_file = (
        config.output_root / f"{config.dataset}_dino_features" / config.model_type / shape / "compression_info.pt"
    )
    os.makedirs(reduction_info_file.parent, exist_ok=True)
    return reduction_info_file


def get_subset_element_features(config: ExtractDinoFeaturesConfig, subset_files: list, get_sample_feature_name_fn):
    subset_dir = get_subset_dir(config)

    # Check how many elements from each file to use
    num_elements_per_file = config.num_elements_for_reduction_matrix // len(subset_files)
    residual_elements = config.num_elements_for_reduction_matrix % len(subset_files)
    print("Number of elements per file: ", num_elements_per_file)
    print("Residual elements: ", residual_elements)

    element_features = torch.zeros(
        (config.num_elements_for_reduction_matrix, config.dino_feature_dim), dtype=torch.float32
    )
    for i, file in enumerate(subset_files):
        denoised_feature = np.load(subset_dir / f"denoised_{get_sample_feature_name_fn(file)}.npy", mmap_mode="r+")
        denoised_feature = torch.from_numpy(denoised_feature).view(-1, config.dino_feature_dim)
        num_elements = num_elements_per_file + 1 if i < residual_elements else num_elements_per_file
        selected_elements = np.random.choice(denoised_feature.shape[0], num_elements, replace=False)
        element_features[i * num_elements_per_file : (i + 1) * num_elements_per_file, :] = denoised_feature[
            selected_elements, :
        ]

    # Perform PCA
    element_features = element_features.to("cuda") if config.pca_use_cuda else element_features

    return element_features


def list_argoverse_files(config: ExtractDinoFeaturesConfig) -> list:
    """
    The files are in the following format:
    dir_path/set_name/sequence_name/sensors/cameras/camera_name/file_name.jpg
    This function lists all the jpg files in the directory of set_name in [train, val, test]
    For the cameras in cameras_to_use
    """

    argoverse_file_list_file = config.output_root / f"{config.dataset}_dino_features" / "argoverse_file_list.txt"

    if argoverse_file_list_file.exists():
        with argoverse_file_list_file.open("r") as f:
            return [line.strip() for line in f]

    def list_files(camera_path):
        return [str(file) for file in camera_path.iterdir() if file.suffix == ".jpg"]

    files = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for set_name in ["train", "val", "test"]:
            set_path = config.argoverse_input_dir / set_name
            for sequence_name in set_path.iterdir():
                for camera_name in ALL_ARGOVERSE_CAMERAS:
                    camera_path = sequence_name / "sensors" / "cameras" / camera_name
                    futures.append(executor.submit(list_files, camera_path))

        for future in futures:
            files.extend(future.result())

    # Randomly shuffle the files
    np.random.shuffle(files)

    # Save the file list
    os.makedirs(config.output_root / f"{config.dataset}_dino_features", exist_ok=True)
    with open(argoverse_file_list_file, "w") as f:
        f.writelines(f"{file}\n" for file in files)

    return files


def get_argoverse_sample_feature_name(file: str):
    sequence_name = file.split("/sensors", maxsplit=1)[0].split("/")[-1]
    camera_name = file.rsplit("/cameras/", maxsplit=1)[-1].split("/")[0]
    file_name = file.rsplit("/", maxsplit=1)[-1].split(".")[0]
    return f"{sequence_name}_{file_name}_{camera_name}"


class ArgoverseInferenceEngine:
    def __init__(self, config: ExtractDinoFeaturesConfig):
        self.config = config

        model, vit, device = _get_model(
            backbone_type=config.backbone_type,
            model_type=config.model_type,
            model_stride=config.model_stride,
            enable_pe=config.enable_pe,
            load_denoiser_from=config.denoiser,
            pos_h=config.pos_h,
            pos_w=config.pos_w,
            use_cuda=config.use_cuda,
        )  # type: ignore

        prep_center = _get_prep(
            image_shape=config.argoverse_center_size,
            crop=config.argoverse_center_crop,
            random_square_crop=None,
            vit=vit,
        )

        prep_other = _get_prep(
            image_shape=config.argoverse_other_size,
            crop=config.argoverse_other_crop,
            random_square_crop=None,
            vit=vit,
        )

        self.model = model
        self.vit = vit
        self.device = device
        self.prep_center = prep_center
        self.prep_other = prep_other

    def infer(self, files: list):
        preps = [self.prep_center if "ring_front_center" in file else self.prep_other for file in files]
        with torch.no_grad():
            denoised_features, raw_features = _forward(self.model, self.vit, files, preps, self.device)
        return denoised_features, raw_features


def main(config: ExtractDinoFeaturesConfig):
    setup()
    print(config)

    if config.dataset == "argoverse":
        print("Listing argoverse files")
        files = list_argoverse_files(config)
        if config.argoverse_cameras_to_use != ALL_ARGOVERSE_CAMERAS:
            files = [file for file in files if any(camera in file for camera in config.argoverse_cameras_to_use)]
        print("Number of files: ", len(files))
        get_sample_feature_name_fn = get_argoverse_sample_feature_name
    else:
        raise NotImplementedError("Dataset not implemented")

    # Initialize the engine
    if config.dataset == "argoverse":
        engine = ArgoverseInferenceEngine(config)
    else:
        raise NotImplementedError("Dataset not implemented")

    # Subset the files to create the PCA reduction matrix
    subset_files = files[: config.num_subset_for_reduction_matrix]
    print("Number of subset files: ", len(subset_files))
    # Infer the features for the subset
    subset_dir = get_subset_dir(config)
    # If we have as many files already in the subset dir, we can use them directly
    if len(list(subset_dir.iterdir())) == len(subset_files) and config.use_if_exists:
        print("Subset files already exist. Skipping inference.")
    else:
        batched_subset_files = [
            subset_files[i : i + config.batch_size] for i in range(0, len(subset_files), config.batch_size)
        ]
        for batched_files in tqdm(batched_subset_files, desc="Processing subset files"):
            denoised_feature, _ = engine.infer(batched_files)
            with ThreadPoolExecutor() as executor:
                futures = []
                for i, file in enumerate(batched_files):
                    out_file = subset_dir / f"denoised_{get_sample_feature_name_fn(file)}.npy"
                    futures.append(executor.submit(np.save, out_file, denoised_feature[i].cpu().numpy()))
                for future in futures:
                    future.result()

    pca_reduction_matrix_file = get_pca_reduction_matrix_file(config)
    element_features = None
    if pca_reduction_matrix_file.exists() and config.use_if_exists:
        print("PCA reduction matrix already exists. Skipping PCA.")
        compression_matrix = torch.load(pca_reduction_matrix_file).to("cuda" if config.pca_use_cuda else "cpu")
    else:
        # Load the features and create the PCA reduction matrix
        element_features = get_subset_element_features(config, subset_files, get_sample_feature_name_fn)
        U, S, compression_matrix = torch.pca_lowrank(element_features, q=config.target_feature_dim, niter=20)
        variances = S**2
        fraction_var_explained = variances / variances.sum()
        print(f"[PCA] fraction_var_explained: \n{fraction_var_explained}")
        print(f"[PCA] fraction_var_explained sum: {fraction_var_explained.sum()}")

        # Save the PCA reduction matrix
        torch.save(compression_matrix, pca_reduction_matrix_file)

    # Compute the compression information
    compression_info_file = get_reduction_info_file(config)
    if compression_info_file.exists() and config.use_if_exists:
        print("Compression information already exists. Skipping.")

    else:
        # Compress the subset files
        compression_matrix_cpu = compression_matrix.cpu()
        for file in tqdm(subset_files, desc="Compressing subset files"):
            subset_file = subset_dir / f"denoised_{get_sample_feature_name_fn(file)}.npy"
            compressed_subset_file = (
                get_compressed_subset_dir(config) / f"compressed_{get_sample_feature_name_fn(file)}.npy"
            )
            if compressed_subset_file.exists() and config.use_if_exists:
                continue
            else:
                denoised_feature = np.load(subset_file, mmap_mode="r+")
                denoised_feature = (
                    torch.from_numpy(denoised_feature)
                    .view(-1, config.dino_feature_dim)
                    .to("cuda" if config.pca_use_cuda else "cpu")
                )
                compressed_feature = torch.mm(denoised_feature, compression_matrix)
                compressed_feature = compressed_feature.view(-1, config.target_feature_dim)
                np.save(compressed_subset_file, compressed_feature.cpu().numpy())

        # Compute the compression information
        subset_compressed_features = []
        for file in subset_files:
            compressed_subset_file = (
                get_compressed_subset_dir(config) / f"compressed_{get_sample_feature_name_fn(file)}.npy"
            )
            compressed_feature = torch.from_numpy(np.load(compressed_subset_file, mmap_mode="r+"))
            subset_compressed_features.append(compressed_feature)
        subset_compressed_features = torch.cat(subset_compressed_features, dim=0)

        compression_info = {
            "compression_matrix": compression_matrix_cpu,
            "compression_min": subset_compressed_features.min(dim=0).values,
            "compression_max": subset_compressed_features.max(dim=0).values,
        }

        for i, dim in enumerate(config.feature_dims_to_rgb):
            rgb_compression_matrix, rgb_min, rgb_max = get_robust_pca(
                subset_compressed_features[:, :dim].to("cuda" if config.pca_use_cuda else "cpu")
            )
            compression_info[f"compression_matrix_{dim}_to_rgb"] = rgb_compression_matrix.cpu()
            compression_info[f"compression_min_{dim}_to_rgb"] = rgb_min.cpu()
            compression_info[f"compression_max_{dim}_to_rgb"] = rgb_max.cpu()

        # Save compression information
        torch.save(compression_info, compression_info_file)
        print("Compression information saved.")

    # Infer the features for all the files
    compressed_features_dir = get_compressed_features_dir(config)
    # Shuffle files
    np.random.shuffle(files)
    # Print first few files
    print("First few files: ", files[:5])

    batched_files = [files[i : i + config.batch_size] for i in range(0, len(files), config.batch_size)]

    for batched_file in tqdm(batched_files, desc="Processing files"):
        out_files = [
            compressed_features_dir / f"{get_sample_feature_name_fn(file)}_compressed_{config.target_feature_dim}.npy"
            for file in batched_file
        ]

        # remove files that already exist from list
        if config.use_if_exists:
            batched_file = [file for file, out_file in zip(batched_file, out_files) if not out_file.exists()]

        # Skip if all files already exist
        if len(batched_file) == 0:
            continue

        denoised_features, _ = engine.infer(batched_file)
        shape = denoised_features.shape
        denoised_features = denoised_features.reshape(-1, shape[-1])
        compressed_features = torch.mm(denoised_features, compression_matrix)
        compressed_features = compressed_features.reshape(shape[0], shape[1], shape[2], config.target_feature_dim)
        compressed_features = compressed_features.cpu().detach().numpy()

        with ThreadPoolExecutor() as executor:
            futures = []
            for i, file in enumerate(batched_file):
                out_file = (
                    compressed_features_dir
                    / f"{get_sample_feature_name_fn(file)}_compressed_{config.target_feature_dim}.npy"
                )
                futures.append(executor.submit(np.save, out_file, compressed_features[i]))
            for future in futures:
                future.result()


def get_robust_pca(
    features: torch.Tensor, m: float = 2.5, remove_first_component: bool = False, output_colors: bool = False
):
    """STOLEN FROM EMERNERF"""
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    rins = colors[fg_mask][s[:, 0] < m, 0]
    gins = colors[fg_mask][s[:, 1] < m, 1]
    bins = colors[fg_mask][s[:, 2] < m, 2]

    rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
    rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    if output_colors:
        return colors
    else:
        return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            ExtractDinoFeaturesConfig,
            description=convert_markup_to_ansi(__doc__),  # type: ignore
        )
    )


if __name__ == "__main__":
    entrypoint()
