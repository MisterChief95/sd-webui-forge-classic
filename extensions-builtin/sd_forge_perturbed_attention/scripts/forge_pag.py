import gradio as gr

from modules import scripts
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling.sampling_function import calc_cond_uncond_batch
from modules.ui_components import InputAccordion


class PerturbedAttentionGuidanceForForge(scripts.Script):
    sorting_priority = 13

    def title(self):
        return "Perturbed Attention Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            with gr.Row():
                scale = gr.Slider(label='Scale', minimum=0.0, maximum=100.0, step=0.1, value=2.5)
                attenuation = gr.Slider(label='Attenuation (linear, % of scale)', minimum=0.0, maximum=100.0, step=0.1,
                                        value=15.0)
            with gr.Row():
                start_step = gr.Slider(label='Start step', minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                end_step = gr.Slider(label='End step', minimum=0.0, maximum=1.0, step=0.01, value=0.4)
            with gr.Row():
                apply_to = gr.Radio(
                    label='Apply To',
                    choices=['Both', 'Base Only', 'Hires Only'],
                    value='Hires Only',
                    visible=not is_img2img,
                    info="Control when PAG is applied during generation"
                )

        self.infotext_fields = [
            (enabled, lambda d: d.get("pagi_enabled", False)),
            (scale, "pagi_scale"),
            (attenuation, "pagi_attenuation"),
            (start_step, "pagi_start_step"),
            (end_step, "pagi_end_step"),
            (apply_to, "pagi_apply_to"),
        ]

        return enabled, scale, attenuation, start_step, end_step, apply_to

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, scale, attenuation, start_step, end_step, apply_to = script_args

        if not enabled:
            return

        # Check if we should skip based on apply_to setting
        is_hires = getattr(p, 'is_hr_pass', False)
        if apply_to == 'Base Only' and is_hires:
            return
        elif apply_to == 'Hires Only' and not is_hires:
            return

        PerturbedAttentionGuidanceForForge.scale = scale
        PerturbedAttentionGuidanceForForge.start_step = start_step
        PerturbedAttentionGuidanceForForge.end_step = end_step
        PerturbedAttentionGuidanceForForge.do_pag = True

        def denoiser_callback(params):
            current_step = params.sampling_step / (params.total_sampling_steps - 1)
            PerturbedAttentionGuidanceForForge.do_pag = (
                current_step >= PerturbedAttentionGuidanceForForge.start_step and
                current_step <= PerturbedAttentionGuidanceForForge.end_step
            )

        on_cfg_denoiser(denoiser_callback)

        unet = p.sd_model.forge_objects.unet.clone()

        def attn_proc(q, k, v, to):
            return v

        def post_cfg_function(args):
            denoised = args["denoised"]

            if PerturbedAttentionGuidanceForForge.scale <= 0.0 or not PerturbedAttentionGuidanceForForge.do_pag:
                return denoised

            model = args["model"]
            cond_denoised = args["cond_denoised"]
            cond = args["cond"]
            sigma = args["sigma"]
            x = args["input"]
            options = args["model_options"].copy()

            new_options = set_model_options_patch_replace(options, attn_proc, "attn1", "middle", 0)

            degraded, _ = calc_cond_uncond_batch(model, cond, None, x, sigma, new_options)

            result = denoised + (cond_denoised - degraded) * PerturbedAttentionGuidanceForForge.scale
            PerturbedAttentionGuidanceForForge.scale -= scale * attenuation / 100.0

            return result

        unet.set_model_sampler_post_cfg_function(post_cfg_function)

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            pagi_enabled=enabled,
            pagi_scale=scale,
            pagi_attenuation=attenuation,
            pagi_start_step=start_step,
            pagi_end_step=end_step,
            pagi_apply_to=apply_to,
        ))

        return

    def postprocess(self, p, processed, *args):
        remove_current_script_callbacks()
