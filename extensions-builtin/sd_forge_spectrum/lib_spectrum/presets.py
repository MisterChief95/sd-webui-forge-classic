import os.path
from json import dump, load
from typing import Final

import gradio as gr

from lib_spectrum import logger

PRESET_FILE: Final[os.PathLike] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets.json")


def to_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")

    return bool(value)


BASE_PARAMS: Final[list] = [float, int, float, int, float, int, float]
PARAMS: Final[list] = [*BASE_PARAMS, to_bool, to_bool, *BASE_PARAMS]
BASE_PARAM_COUNT: Final[int] = len(BASE_PARAMS)


class PresetManager:
    presets: dict[str, list] = None

    @classmethod
    def load_presets(cls):
        if cls.presets is not None:
            return

        if not os.path.isfile(PRESET_FILE):
            with open(PRESET_FILE, "w+", encoding="utf-8") as json_file:
                dump({}, json_file)

            logger.debug("Creating new empty Presets...")
            cls.presets = {}
            return

        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as json_file:
                cls.presets = load(json_file)
        except Exception:
            logger.error("Failed to load Presets...")
            cls.presets = {}
        else:
            logger.debug("Loaded Presets...")

    @classmethod
    def list_preset(cls) -> list[str]:
        return list(cls.presets.keys())

    @classmethod
    def get_preset(cls, preset_name: str) -> list:
        if (preset := cls.presets.get(preset_name, None)) is None:
            logger.error(f'Preset "{preset_name}" was not found...')
            return [gr.skip()] * len(PARAMS)

        if isinstance(preset, dict):
            base_preset = preset.get("base", [])
            hires_preset = preset.get("hires", base_preset)
            if len(base_preset) != BASE_PARAM_COUNT or len(hires_preset) != BASE_PARAM_COUNT:
                logger.error(f'Preset "{preset_name}" has an unsupported format...')
                return [gr.skip()] * len(PARAMS)

            preset = [
                *[obj(val) for obj, val in zip(BASE_PARAMS, base_preset)],
                to_bool(preset.get("apply_to_hires", True)),
                to_bool(preset.get("override_hires", False)),
                *[obj(val) for obj, val in zip(BASE_PARAMS, hires_preset)],
            ]
        elif len(preset) == len(BASE_PARAMS):
            base_preset = [obj(val) for obj, val in zip(BASE_PARAMS, preset)]
            preset = [*base_preset, True, False, *base_preset]
        elif len(preset) == len(PARAMS):
            preset = [obj(val) for obj, val in zip(PARAMS, preset)]
        else:
            logger.error(f'Preset "{preset_name}" has an unsupported format...')
            return [gr.skip()] * len(PARAMS)

        return [gr.update(value=val) for val in preset]

    @classmethod
    def save_preset(cls, preset_name: str, *args) -> list[str]:
        if preset_name is None or not preset_name.strip():
            logger.error("Invalid Preset Name...")
            return gr.skip()

        if len(args) != len(PARAMS):
            logger.error("Invalid Spectrum preset data...")
            return gr.skip()

        base_args = args[:BASE_PARAM_COUNT]
        apply_to_hires = args[BASE_PARAM_COUNT]
        override_hires = args[BASE_PARAM_COUNT + 1]
        hires_args = args[BASE_PARAM_COUNT + 2:]

        cls.presets.update({
            preset_name: {
                "base": [*base_args],
                "apply_to_hires": apply_to_hires,
                "override_hires": override_hires,
                "hires": [*hires_args],
            }
        })

        with open(PRESET_FILE, "w", encoding="utf-8") as json_file:
            dump(cls.presets, json_file)

        logger.info(f'Preset "{preset_name}" Saved!')
        return gr.update(choices=cls.list_preset())

    @classmethod
    def delete_preset(cls, preset_name: str) -> list[str]:
        if preset_name not in cls.presets:
            logger.error(f'Preset "{preset_name}" was not found...')
            return gr.skip()

        del cls.presets[preset_name]

        with open(PRESET_FILE, "w", encoding="utf-8") as json_file:
            dump(cls.presets, json_file)

        logger.info(f'Preset "{preset_name}" Deleted!')
        return gr.update(value=None, choices=cls.list_preset())
