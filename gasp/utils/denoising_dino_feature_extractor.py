import re
import types
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import timm
import timm.data
import torch
from PIL import Image

try:
    from timm.layers import resample_abs_pos_embed
except ImportError:
    print("resample_abs_pos_embed not found in timm.layers. Using local version.")

from timm.models.eva import Eva
from timm.models.vision_transformer import Block, Mlp, VisionTransformer
from torch import nn
from torchvision import transforms
from tqdm import trange

# We have played with these models, Feel free to add more models to the list.
MODEL_LIST = [
    # DINOv1
    "vit_small_patch8_224.dino",
    "vit_small_patch16_224.dino",
    "vit_base_patch8_224.dino",
    "vit_base_patch16_224.dino",
    # DINOv2
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    # DINOv2 + register
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_giant_patch14_reg4_dinov2.lvd142m",
    # MAE
    "vit_base_patch16_224.mae",
    "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    # CLIP
    "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "vit_base_patch16_clip_224.openai",
    # EVA
    "eva02_base_patch16_clip_224.merged2b",
    # DEiT-III
    "deit3_base_patch16_224.fb_in1k",
    # Auto-auged supervised ViT:
    "vit_base_patch16_384.augreg_in21k_ft_in1k",
    # commented out for simplicity. Do not use these models for now
    # it's a bit annoying to hack the intermediate layers for these models
    # however, in our informal early experiments, these models all exhibit
    # similar artifacts as the models above.
    # the artifacts in SAM are similar to the ones in MAE, while the
    # artifacts in iJEPGA are similar to the ones in DeiT.
    # SAM
    # "samvit_base_patch16.sa1b",
    # "samvit_large_patch16.sa1b",
    # "samvit_huge_patch16.sa1b",
    # ijepga
    # "vit_huge_patch14_224_ijepa.in1k",
]


class ViTWrapper(nn.Module):
    def __init__(
        self,
        model_identifier: str = "vit_base_patch14_dinov2.lvd142m",
        stride: int = 7,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        **kwargs,
    ):
        super().__init__()
        # comment out the following line to test the models not in the list
        assert model_identifier in MODEL_LIST, f"Model type {model_identifier} not tested yet."
        self.model_identifier = model_identifier
        self.stride = stride
        self.patch_size = int(re.search(r"patch(\d+)", model_identifier).group(1))
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.model, self.transformation = self.create_model(model_identifier, **kwargs)
        # overwrite the stride size
        if stride != self.model.patch_embed.proj.stride[0]:
            self.model.patch_embed.proj.stride = [stride, stride]

            def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
                """Get grid (feature) size for given image size taking account of dynamic padding.
                NOTE: must be torchscript compatible so using fixed tuple indexing
                """
                return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
                    img_size[1] - self.patch_size[1]
                ) // self.proj.stride[1] + 1

            self.model.patch_embed.dynamic_feat_size = types.MethodType(dynamic_feat_size, self.model.patch_embed)

    @property
    def n_output_dims(self) -> int:
        return self.model.pos_embed.shape[-1]

    @property
    def num_blocks(self) -> int:
        return len(self.model.blocks)

    @property
    def last_layer_index(self) -> int:
        return self.num_blocks - 1

    def create_model(self, model_identifier: str, **kwargs) -> Tuple[Union[VisionTransformer, Eva], transforms.Compose]:
        model = timm.create_model(
            model_identifier,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=self.dynamic_img_size,
            dynamic_img_pad=self.dynamic_img_pad,
            **kwargs,
        )
        # Different models have different data configurations
        # e.g., their training resolution, normalization, etc, are different
        data_config = timm.data.resolve_model_data_config(model=model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return model, transforms

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
        reshape: bool = True,
        return_prefix_tokens: bool = False,
        norm: bool = True,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Intermediate layer accessor inspired by DINO / DINOv2 interface.
        Args:
            x: Input tensor.
            n: Take last n blocks if int, all if None, select matching indices if sequence
            reshape: Whether to reshape the output.
        """
        return self.model.forward_intermediates(
            x,
            n,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            output_fmt="NCHW" if reshape else "NLC",
            intermediates_only=True,
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class Denoiser(nn.Module):
    def __init__(
        self,
        noise_map_height: int = 37,
        noise_map_width: int = 37,
        feat_dim: int = 768,
        vit: ViTWrapper = None,
        enable_pe: bool = True,
        num_blocks: int = 1,
    ):
        super().__init__()
        self.vit = vit
        self.denoiser = Block(
            dim=feat_dim,
            num_heads=feat_dim // 64,
            mlp_ratio=4,
            qkv_bias=True,
            qk_norm=False,
            init_values=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            mlp_layer=Mlp,
        )
        if num_blocks > 1:
            self.denoiser = nn.Sequential(
                *[
                    Block(
                        dim=feat_dim,
                        num_heads=feat_dim // 64,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_norm=False,
                        init_values=None,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        act_layer=nn.GELU,
                        mlp_layer=Mlp,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.pos_embed = None
        if enable_pe:
            seq_len = noise_map_height * noise_map_width
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, feat_dim) * 0.02)
        if self.vit is not None:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(
        self,
        x,
        return_dict=False,
        return_channel_first=False,
        return_class_token=False,
        norm=True,
    ):
        class_tokens = None
        if self.vit is not None:
            with torch.no_grad():
                # (B, C, H, W)
                vit_outputs = self.vit.get_intermediate_layers(
                    x,
                    n=[self.vit.last_layer_index],
                    return_prefix_tokens=return_class_token,
                    norm=norm,
                )
                if return_class_token:
                    vit_outputs = vit_outputs[-1]
                    class_tokens = vit_outputs[1][:, 0]
                original_feats = vit_outputs[0].permute(0, 2, 3, 1)
                x = original_feats.detach().clone()
        else:
            original_feats = x.clone()
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed(self.pos_embed, (h, w), num_prefix_tokens=0)
        x = self.denoiser(x)
        x = x.reshape(b, h, w, c)
        if return_channel_first:
            x = x.permute(0, 3, 1, 2)
        if return_dict:
            return {
                "denoised_feats": x,
                "original_feats": original_feats.detach(),
                "class_tokens": class_tokens.detach() if class_tokens is not None else None,
            }
        if return_class_token:
            assert class_tokens is not None
            return x, class_tokens
        return x


class MyCrop(torch.nn.Module):
    def __init__(self, crop: Tuple[int, int, int, int]):
        super(MyCrop, self).__init__()
        self.crop = crop

    def __call__(self, img):
        c = self.crop
        return img[:, c[0] : -c[2] or None, c[3] : -c[1] or None]


def _get_device(use_cuda: bool = True) -> torch.device:
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


def _get_model(
    backbone_type: str = "DeViT",
    model_type: str = "base",
    pre_training_name: str = "dinov2",
    model_stride: int = 7,
    enable_pe: bool = True,
    load_denoiser_from: Union[Path, None] = None,
    pos_h: int = 37,
    pos_w: int = 37,
    use_cuda: bool = True,
) -> Union[ViTWrapper, Denoiser]:
    device = _get_device(use_cuda)
    if backbone_type == "fbdino":
        BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")

        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        vit = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name).eval().to(device)

    elif backbone_type == "DeViT":
        if pre_training_name == "dinov2":
            backbone_archs = {
                "small": "vit_small_patch14_dinov2.lvd142m",
                "v1": "vit_small_patch14_dinov2.lvd142m",
                "base": "vit_base_patch14_dinov2.lvd142m",
                "large": "vit_large_patch14_dinov2.lvd142m",
                "giant": "vit_giant_patch14_dinov2.lvd142m",
            }
        elif pre_training_name == "clip":
            backbone_archs = {
                "base": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
            }
        backbone_name = backbone_archs[model_type]
        vit = ViTWrapper(model_identifier=backbone_name, stride=model_stride).eval().to(device)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    if load_denoiser_from is not None:
        denoiser = Denoiser(
            noise_map_height=pos_h,
            noise_map_width=pos_w,
            feat_dim=vit.n_output_dims,
            vit=vit,
            enable_pe=enable_pe,
        )

        ckpt = torch.load(load_denoiser_from)

        if "denoiser" in ckpt:
            msg = denoiser.load_state_dict(ckpt["denoiser"], strict=False)
            model = denoiser
            print(f"!!! Denoiser loaded from {load_denoiser_from} !!!")

        elif "model" in ckpt:
            msg = vit.load_state_dict(ckpt["model"], strict=False)
            model = vit
            print(f"!!! Distilled model loaded from {load_denoiser_from} !!!")
        else:
            raise ValueError("Invalid checkpoint format")

        missing_keys = msg.missing_keys
        unwanted_keys = msg.unexpected_keys
        missing_keys = {k for k in missing_keys if "vit" not in k}
        print(f"Missing keys: {missing_keys}, unexpected keys: {unwanted_keys}")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)

    else:
        print("!!! No denoiser loaded, using backbone only !!!")
        model = vit

    return model, vit, device


def _get_prep(
    image_shape: Tuple[int, int] = (640, 960),
    crop: Tuple[int, int, int, int] = (0, 0, 0, 0),
    random_square_crop: Union[int, None] = None,
    vit: ViTWrapper = None,
) -> transforms.Compose:
    mytransforms = [
        transforms.ToTensor(),
        MyCrop(crop),
        transforms.Resize(image_shape),
        # transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
        #    std=[0.229, 0.224, 0.225],
        # ),
        vit.transformation.transforms[-1],
    ]
    if random_square_crop is not None:
        mytransforms.append(transforms.RandomCrop(random_square_crop))

    prep = transforms.Compose(mytransforms)

    return prep


def _forward(model, vit, input_image_filepath, prep, device):
    if isinstance(input_image_filepath, list):
        image = [Image.open(f).convert("RGB") for f in input_image_filepath]
        image = [p(i).unsqueeze(0).to(device) for p, i in zip(prep, image)]
        image = torch.cat(image, dim=0)
    else:
        image = Image.open(input_image_filepath).convert("RGB")
        image = prep(image).unsqueeze(0).to(device)

    if isinstance(model, ViTWrapper):
        denoised_features = vit.get_intermediate_layers(
            image,
            n=[vit.last_layer_index],
            reshape=True,
            norm=True,
        )[-1].permute(0, 2, 3, 1)

        raw_features = None

    elif isinstance(model, Denoiser):
        out_dict = model(image, return_dict=True)
        denoised_features = out_dict["denoised_feats"]
        raw_features = out_dict["original_feats"]

    else:
        raise ValueError("Invalid model type")

    return denoised_features, raw_features


def extract_and_save_denoised_features(
    input_img_path_list: List[str],
    saved_feat_path_list: List[str],
    backbone_type: str = "DeViT",
    model_type: str = "small",
    model_stride: int = 7,
    load_denoiser_from: Union[Path, None] = None,
    save_non_denoised_features: bool = False,
    enable_pe: bool = True,
    image_shape: Tuple[int, int] = (640, 960),
    pos_h: int = 37,
    pos_w: int = 37,
    crop: Tuple[int, int, int, int] = (0, 0, 0, 0),
    use_cuda: bool = True,
    random_square_crop: Union[int, None] = None,
    force_overwrite: bool = False,
):
    model, vit, device = _get_model(
        backbone_type=backbone_type,
        model_type=model_type,
        model_stride=model_stride,
        enable_pe=enable_pe,
        load_denoiser_from=load_denoiser_from,
        pos_h=pos_h,
        pos_w=pos_w,
        use_cuda=use_cuda,
    )

    prep = _get_prep(
        image_shape=image_shape,
        crop=crop,
        random_square_crop=random_square_crop,
        vit=vit,
    )

    with torch.no_grad():
        for i in trange(0, len(input_img_path_list), desc="Extracting features"):
            input_image_filepath = input_img_path_list[i]
            output_feature_filepath = saved_feat_path_list[i]

            if output_feature_filepath.exists() and not force_overwrite:
                continue

            denoised_features, raw_features = _forward(model, vit, input_image_filepath, prep, device)
            np.save(output_feature_filepath, denoised_features.cpu().numpy())

            if save_non_denoised_features and raw_features is not None:
                raw_output_feature_filepath = output_feature_filepath.parent / Path(
                    f"{output_feature_filepath.stem}_raw.npy"
                )
                np.save(raw_output_feature_filepath, raw_features.cpu().numpy())


def extract_and_compress_and_save_denoised_features(
    input_img_path_list: List[str],
    saved_feat_path_list: List[str],
    backbone_type: str = "DeViT",
    model_type: str = "small",
    model_stride: int = 7,
    load_denoiser_from: Union[Path, None] = None,
    save_non_denoised_features: bool = False,
    enable_pe: bool = True,
    image_shape: Tuple[int, int] = (640, 960),
    pos_h: int = 37,
    pos_w: int = 37,
    crop: Tuple[int, int, int, int] = (0, 0, 0, 0),
    use_cuda: bool = True,
    random_square_crop: Union[int, None] = None,
    force_overwrite: bool = False,
):
    model, vit, device = _get_model(
        backbone_type=backbone_type,
        model_type=model_type,
        model_stride=model_stride,
        enable_pe=enable_pe,
        load_denoiser_from=load_denoiser_from,
        pos_h=pos_h,
        pos_w=pos_w,
        use_cuda=use_cuda,
    )

    prep = _get_prep(
        image_shape=image_shape,
        crop=crop,
        random_square_crop=random_square_crop,
        vit=vit,
    )

    compression_matrix_path = saved_feat_path_list[0].parent / Path("compression_matrix.npy")
    if not compression_matrix_path.exists():
        raise ValueError(f"Compression matrix not found: {compression_matrix_path}. Make sure to create it first.")
    else:
        print(f"Using existing compression matrix: {compression_matrix_path}")

    compression_matrix = np.load(compression_matrix_path)
    compression_matrix = torch.from_numpy(compression_matrix).to(device)
    target_feature_dim = compression_matrix.shape[1]

    print(f"Target feature dim: {target_feature_dim}")

    with torch.no_grad():
        for i in trange(0, len(input_img_path_list), desc="Extracting and compressing features"):
            input_image_filepath = input_img_path_list[i]
            output_feature_filepath = saved_feat_path_list[i]

            if output_feature_filepath.exists() and not force_overwrite:
                continue

            features, _ = _forward(model, vit, input_image_filepath, prep, device)

            features_shape = features.squeeze(0).shape
            features = features.reshape(-1, features.shape[-1])
            compressed_features = torch.mm(features, compression_matrix)
            compressed_features = compressed_features.reshape(features_shape[0], features_shape[1], target_feature_dim)
            compressed_features = compressed_features.cpu().detach().numpy()

            np.save(output_feature_filepath, compressed_features)
