import html
import os

from modules import sd_models, shared, ui_extra_networks
from modules.ui_extra_networks_checkpoints_user_metadata import CheckpointUserMetadataEditor


class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__("Checkpoints")

        self.allow_prompt = False

    def refresh(self):
        shared.refresh_checkpoints()

    def create_item(self, name, index=None, enable_filter=True):
        checkpoint: sd_models.CheckpointInfo = sd_models.checkpoint_aliases.get(name)
        if checkpoint is None:
            return

        path = os.path.splitext(checkpoint.filename)[0]
        search_terms = [self.search_terms_from_path(checkpoint.filename)]
        if checkpoint.sha256:
            search_terms.append(checkpoint.sha256)

        user_metadata: dict = ui_extra_networks.extra_networks.get_user_metadata(checkpoint.filename)
        preferred_modules: list[str] = user_metadata.get("vae_te", [])

        return {
            "name": checkpoint.name_for_extra,
            "filename": checkpoint.filename,
            "shorthash": checkpoint.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_terms": search_terms,
            "onclick": html.escape(f"return selectCheckpoint({ui_extra_networks.quote_js(name)}, {ui_extra_networks.quote_js(preferred_modules)})"),
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": checkpoint.metadata,
            "sort_keys": {"default": index, **self.get_sort_keys(checkpoint.filename)},
        }

    def list_items(self):
        # instantiate a list to protect against concurrent modification
        names = list(sd_models.checkpoints_list)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [v for v in (*shared.cmd_opts.ckpt_dirs, sd_models.model_path) if os.path.isdir(str(v))]

    def create_user_metadata_editor(self, ui, tabname):
        return CheckpointUserMetadataEditor(ui, tabname, self)
