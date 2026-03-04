from dataclasses import dataclass


@dataclass
class LoggingConfig:
    img_features: bool = False
    img_depths: bool = False
    img_lifted_features: bool = False
    img_lifted_processed_features: bool = False
    img_lifted_bev_occupancy: bool = False
    img_temporally_fused_bev_features: bool = False
    bev_features: bool = True
    offsets: bool = False
    masked_features: bool = True
