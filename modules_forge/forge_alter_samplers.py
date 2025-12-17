import inspect
import logging
from typing import Callable

import k_diffusion.sampling

from modules import sd_samplers_common, sd_samplers_kdiffusion


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name, **kwargs):
        sampler_function: Callable = getattr(k_diffusion.sampling, f"sample_{sampler_name}", None)
        if sampler_function is None:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        # Store additional kwargs like cfg_pp for later use
        self.sampler_kwargs = kwargs
        self.is_cfg_pp = sampler_name.endswith("_cfg_pp") or kwargs.get("cfg_pp", False)
        super().__init__(sampler_function, sd_model)

    def initialize(self, p) -> dict:
        extra_params_kwargs = super().initialize(p)

        # Add any additional sampler-specific kwargs (like cfg_pp)
        parameters = inspect.signature(self.func).parameters
        for key, value in self.sampler_kwargs.items():
            if key in parameters:
                extra_params_kwargs[key] = value

        return extra_params_kwargs

    def sample(self, p, *args, **kwargs):
        if self.is_cfg_pp and p.cfg_scale > 2.0:
            logging.warning("CFG between 1.0 ~ 2.0 is recommended when using CFG++ samplers")
        return super().sample(p, *args, **kwargs)

    def sample_img2img(self, p, *args, **kwargs):
        if self.is_cfg_pp and p.cfg_scale > 2.0:
            logging.warning("CFG between 1.0 ~ 2.0 is recommended when using CFG++ samplers")
        return super().sample_img2img(p, *args, **kwargs)


def build_constructor(sampler_key: str, **kwargs) -> Callable:
    def constructor(model):
        return AlterSampler(model, sampler_key, **kwargs)

    return constructor


def create_alter_sampler(sampler_name: str, sampler_key: str, copy_params_name: str | None = None, **kwargs) -> "sd_samplers_common.SamplerData":
    config = {}
    if copy_params_name:
        for name, _, _, params in sd_samplers_kdiffusion.samplers_k_diffusion:
            if name == copy_params_name:
                config = params.copy()
                break

    return sd_samplers_common.SamplerData(sampler_name, build_constructor(sampler_key=sampler_key, **kwargs), [sampler_key], config)


samplers_data_alter = [
    create_alter_sampler("DPM++ 2M CFG++", "dpmpp_2m_cfg_pp", copy_params_name="DPM++ 2M"),
    create_alter_sampler("Euler a CFG++", "euler_ancestral_cfg_pp", copy_params_name="Euler a"),
    create_alter_sampler("Euler CFG++", "euler_cfg_pp", copy_params_name="Euler"),
    create_alter_sampler("Res Multistep a", "res_multistep", eta=0.),
    create_alter_sampler("Res Multistep CFG++", "res_multistep", eta=0., cfg_pp=True),
    create_alter_sampler("Res Multistep a CFG++", "res_multistep", eta=1., cfg_pp=True),
    create_alter_sampler("Gradient Estimation CFG++", "gradient_estimation", cfg_pp=True),
    create_alter_sampler("Exponential Heun2 x0", "seeds_2", copy_params_name="Seeds2", s_noise=0., eta=0., solver_type="phi_2", r=1.),
    create_alter_sampler("Exponential Heun2 x0 SDE", "seeds_2", copy_params_name="Seeds2", s_noise=1., eta=1., solver_type="phi_2", r=1.),
]
