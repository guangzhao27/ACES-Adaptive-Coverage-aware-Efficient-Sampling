
import os

import torch

from .fourier_mlp import FourierFeatureMLP
from .siren import ModulatedSiren, Siren


def create_inr_instance(cfg, input_dim=1, output_dim=1, device="cuda"):
    device = torch.device(device)
    if cfg.inr.model_type == "siren":
        inr = ModulatedSiren(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            w0=cfg.inr.w0,
            w0_initial=cfg.inr.w0,
            use_bias=True,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            use_latent=cfg.inr.use_latent,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
            last_activation=cfg.inr.last_activation,
        ).to(device)

    elif cfg.inr.model_type == "single_image_siren":
        inr = Siren(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            w0=cfg.inr.w0,
            w0_initial=cfg.inr.w0,
            use_bias=True,
        ).to(device)

    elif cfg.inr.model_type in {"single_image_fourier_mlp", "fourier_feature_mlp"}:
        inr = FourierFeatureMLP(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            mapping_size=getattr(cfg.inr, "fourier_mapping_size", 256),
            scale=getattr(cfg.inr, "fourier_scale", 10.0),
            include_input=getattr(cfg.inr, "fourier_include_input", True),
            activation=getattr(cfg.inr, "fourier_activation", "relu"),
        ).to(device)

    else:
        raise NotImplementedError(f"No corresponding class for {cfg.inr.model_type}")

    return inr


def load_inr_model(
    run_dir, run_name, data_to_encode, input_dim=1, output_dim=1, device="cuda"
):
    inr_train = torch.load(os.path.join(run_dir, run_name+'.pt'), weights_only=False)

    inr_state_dict = inr_train["inr"]
    cfg = inr_train["cfg"]
    alpha = inr_train["alpha"]

    inr = create_inr_instance(cfg, input_dim, output_dim, device)
    inr.load_state_dict(inr_state_dict)
    inr.eval()

    return inr, alpha
