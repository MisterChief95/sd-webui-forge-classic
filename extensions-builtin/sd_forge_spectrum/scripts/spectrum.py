import gradio as gr
from lib_spectrum.forecaster import SpectrumNode
from lib_spectrum.presets import PresetManager

from modules import scripts, shared
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

PresetManager.load_presets()

PARAM_KEYS = [
    "spec_w",
    "spec_m",
    "spec_lam",
    "spec_window_size",
    "spec_flex_window",
    "spec_warmup_steps",
    "spec_stop_caching_step",
]
PARAM_HR_KEYS = [f"spec_hr_{key.removeprefix('spec_')}" for key in PARAM_KEYS]
PARAM_COUNT = len(PARAM_KEYS)


class SpectrumForForge(scripts.Script):
    sorting_priority = 2026

    def title(self):
        return "Spectrum Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def create_spectrum_controls(self):
        with gr.Row():
            w = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Prediction Weighting",
                info="higher = long-term trend ; lower = short-term changes",
            )
            m = gr.Slider(
                minimum=1,
                maximum=8,
                value=6,
                step=1,
                label="Polynomial Degree",
                info="higher = complex & subtle patterns ; lower = stable & faster",
            )
        with gr.Row():
            lam = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.5,
                step=0.05,
                label="Regularization",
                info="higher = reduce overfitting ; lower = fit more data",
            )
            window_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=2,
                step=1,
                label="Cache Window",
                info="higher = skip more steps ; lower = slower but more accurate",
            )
        flex_window = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.0,
            step=0.05,
            label="Window Growth",
            info="higher = more speed & less accurate ; lower = more consistent accuracy but less speed gain",
        )
        with gr.Row():
            warmup_steps = gr.Slider(
                minimum=0,
                maximum=20,
                value=6,
                step=1,
                label="Warmup Steps",
                info="Run the full model before caching starts",
            )
            stop_caching_step = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Stop Caching Step",
                info="Run the full model for the last few steps",
            )

        return (w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step)

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            apply_to_hires = gr.Checkbox(
                value=True,
                label="Apply to Hires Fix",
                info="Run Spectrum during the hires fix pass.",
            )

            base_args = self.create_spectrum_controls()

            with InputAccordion(False, label="Override Hires Fix Settings") as override_hires:
                hires_args = self.create_spectrum_controls()

            preset_args = (*base_args, apply_to_hires, override_hires, *hires_args)

            with gr.Accordion("Presets", open=False):
                _preset = gr.Dropdown(
                    value=None,
                    label="Preset Name",
                    choices=PresetManager.list_preset(),
                    allow_custom_value=True,
                )
                with gr.Row():
                    _load = gr.Button("Apply Preset", variant="secondary")
                    _save = gr.Button("Save Preset", variant="primary")
                    _del = gr.Button("Delete Preset", variant="stop")

                for comp in (_preset, _load, _save, _del):
                    comp.do_not_save_to_config = True

                _load.click(
                    fn=lambda name: PresetManager.get_preset(name),
                    inputs=[_preset],
                    outputs=[*preset_args],
                    queue=False,
                )
                _save.click(
                    fn=lambda *args: PresetManager.save_preset(*args),
                    inputs=[_preset, *preset_args],
                    outputs=[_preset],
                    queue=False,
                )
                _del.click(
                    fn=lambda name: PresetManager.delete_preset(name),
                    inputs=[_preset],
                    outputs=[_preset],
                    queue=False,
                )

        self.infotext_fields = [
            *[PasteField(comp, key) for comp, key in zip(base_args, PARAM_KEYS)],
            PasteField(apply_to_hires, "spec_apply_hr"),
            PasteField(override_hires, "spec_hr_override"),
            *[PasteField(comp, key) for comp, key in zip(hires_args, PARAM_HR_KEYS)],
        ]
        self.paste_field_names = [field.label for field in self.infotext_fields]

        return [enable, *base_args, apply_to_hires, override_hires, *hires_args]

    def process_before_every_sampling(self, p, enable: bool, *args, **kwargs):
        if not enable:
            return

        if shared.opts.skip_early_cond > 0.0 or shared.opts.s_min_uncond > 0.0:
            print('Spectrum does not support "Ignore/Skip Negative Prompt" optimizations...')
            return

        base_args = args[:PARAM_COUNT]
        if len(args) >= PARAM_COUNT + 2:
            apply_to_hires = args[PARAM_COUNT]
            override_hires = args[PARAM_COUNT + 1]
            hires_args = args[PARAM_COUNT + 2:]
        else:
            apply_to_hires = True
            override_hires = False
            hires_args = base_args

        is_hr_pass = getattr(p, "is_hr_pass", False)
        if is_hr_pass and not apply_to_hires:
            return

        spectrum_args = hires_args if is_hr_pass and override_hires else base_args
        steps = (p.hr_second_pass_steps or p.steps) if is_hr_pass else p.steps

        unet = p.sd_model.forge_objects.unet
        unet = SpectrumNode.patch(unet, steps, *spectrum_args)
        p.sd_model.forge_objects.unet = unet

        for k, v in zip(PARAM_KEYS, base_args):
            p.extra_generation_params[k] = v

        if getattr(p, "enable_hr", False):
            p.extra_generation_params["spec_apply_hr"] = apply_to_hires

        if getattr(p, "enable_hr", False) and apply_to_hires and override_hires:
            p.extra_generation_params["spec_hr_override"] = override_hires
            for k, v in zip(PARAM_HR_KEYS, hires_args):
                p.extra_generation_params[k] = v
