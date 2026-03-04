import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tyro
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from gasp.config.method_configs import convert_markup_to_ansi

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
class GenerateDinov3ReductionConfig:
    # Data i/o
    output_root: Path = Path("/outputs/feature_reductions")
    """The output directory."""
    argoverse_input_dir: Path = Path("/datasets/av2")
    """The argoverse dataset directory"""

    datasets: Tuple[str, ...] = ("argoverse",)
    """The dataset to use. 'argoverse' or 'ZOD' or 'nuscenes' """
    use_cuda: bool = True
    """Whether to use cuda"""

    feature_extractor_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    """The model to use for feature extraction."""

    num_images_to_use: int = 10  # 4_500
    """The number of images to use for feature extraction."""

    features_per_image: int = 1_000
    """The number of features per image."""

    target_feature_dims: Tuple[int, ...] = (3, 8, 16, 32, 64, 128)
    """The target feature dimensions to use for feature extraction."""

    vit_layers_to_use: Tuple[int, ...] = (7, 9, 11, 12)
    """The layers to use for feature extraction."""

    reduction_types: Tuple[str, ...] = ("pca",)
    """The reduction types to use for feature extraction."""

    @property
    def model_name(self) -> str:
        return self.feature_extractor_model.split("/")[-1].replace("-", "_")


def list_argoverse_files(config: GenerateDinov3ReductionConfig) -> list:
    """
    The files are in the following format:
    dir_path/set_name/sequence_name/sensors/cameras/camera_name/file_name.jpg
    This function lists all the jpg files in the directory of set_name in [train, val, test]
    For the cameras in cameras_to_use
    """

    argoverse_file_list_file = config.output_root / f"{config.model_name}_features" / "argoverse_file_list.txt"

    if argoverse_file_list_file.exists():
        with argoverse_file_list_file.open("r") as f:
            return [line.strip() for line in f]

    def list_files(camera_path):
        return [str(file) for file in camera_path.iterdir() if file.suffix == ".jpg"]

    print("Listing argoverse files")

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
    os.makedirs(argoverse_file_list_file.parent, exist_ok=True)
    with open(argoverse_file_list_file, "w") as f:
        f.writelines(f"{file}\n" for file in files)

    return files


def read_image(file: str, use_cuda: bool) -> np.ndarray:
    image = torch.from_numpy(np.array(Image.open(file).convert("RGB"), dtype=np.float32))
    image = image.unsqueeze(0).to(torch.device("cuda" if use_cuda else "cpu"))
    image = image.permute(0, 3, 1, 2)
    return image


def main(config: GenerateDinov3ReductionConfig):
    import numpy as np  # type: ignore

    setup()
    print(config)

    files = []
    if "argoverse" in config.datasets:
        files += list_argoverse_files(config)[:250_000]

    np.random.shuffle(files)
    files = files[: config.num_images_to_use]

    processor = AutoImageProcessor.from_pretrained(config.feature_extractor_model, do_resize=False, do_rescale=True)
    model = AutoModel.from_pretrained(
        config.feature_extractor_model,
        device_map=None,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if config.use_cuda:
        model.to(torch.device("cuda"))
    else:
        model.to(torch.device("cpu"))

    num_layers = len(config.vit_layers_to_use)
    features_per_image = config.features_per_image
    feature_dim = 768
    num_files = len(files)
    num_features = features_per_image * num_files
    all_features = torch.zeros(
        (num_layers, num_features, feature_dim),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    for i, file in tqdm(enumerate(files), desc="Extracting features"):
        image = read_image(file, config.use_cuda)
        h_feat, w_feat = image.shape[2] // 16, image.shape[3] // 16
        inputs = processor(image, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)

        hidden_states = [outputs.hidden_states[layer_idx][:, 5:, :] for layer_idx in config.vit_layers_to_use]
        hidden_states = torch.cat(hidden_states, dim=0)

        # Select features
        features_in_image = h_feat * w_feat
        selected_feature_indices = np.random.choice(features_in_image, features_per_image, replace=False)
        selected_features = hidden_states[:, selected_feature_indices, :].cpu()
        all_features[:, i * features_per_image : (i + 1) * features_per_image, :] = selected_features

    del model
    del processor
    torch.cuda.empty_cache()

    output_info = {}
    for i, layer_idx in enumerate(config.vit_layers_to_use):
        layer_features = all_features[i, :, :]

        if config.use_cuda:
            layer_features = layer_features.to("cuda")

        layer_info = {}

        mean_features_768 = layer_features.mean(dim=0)
        std_features_768 = layer_features.std(dim=0)

        layer_info["dim_768"] = {
            "mean": mean_features_768.cpu().numpy(),
            "std": std_features_768.cpu().numpy(),
            "pca_compression_matrix": None,
        }

        layer_features -= mean_features_768
        layer_features /= std_features_768 + 1e-8

        for target_feature_dim in config.target_feature_dims:
            if "pca" in config.reduction_types:
                U, S, compression_matrix = torch.pca_lowrank(layer_features, q=target_feature_dim, niter=20)
                variances = S**2
                fraction_var_explained = variances / variances.sum()
                rounded_fve = [round(float(x), 2) for x in fraction_var_explained.cpu().numpy()]
                print(f"[Layer {layer_idx}, PCA to {target_feature_dim}] fraction_var_explained: {rounded_fve}")
                print(
                    f"[Layer {layer_idx}, PCA to {target_feature_dim}] fraction_var_explained sum: {round(float(fraction_var_explained.sum()), 2)}"
                )

                compressed_features = layer_features @ compression_matrix
                mean_features_compressed = compressed_features.mean(dim=0)
                std_features_compressed = compressed_features.std(dim=0)
                layer_info[f"dim_{target_feature_dim}"] = {
                    "pca_mean": mean_features_compressed.cpu().numpy(),
                    "pca_std": std_features_compressed.cpu().numpy(),
                    "pca_compression_matrix": compression_matrix.cpu().numpy(),
                }

        output_info[f"layer_{layer_idx}"] = layer_info

    torch.save(output_info, config.output_root / f"{config.model_name}_features" / "reduction_info.pt")
    print("Reduction information saved.")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            GenerateDinov3ReductionConfig,
            description=convert_markup_to_ansi(__doc__),  # type: ignore
        )
    )


def setup():
    torch.set_float32_matmul_precision("medium")  # heighest, high, medium


if __name__ == "__main__":
    entrypoint()
