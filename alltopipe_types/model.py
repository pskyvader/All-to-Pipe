import os
import folder_paths
import comfy.sd
import comfy.model_patcher


class Model:
    def __init__(self, name: str, subfolder: str, clip_skip: int) -> None:
        self.name: str = name
        self.subfolder: str = subfolder
        self.clip_skip: int = clip_skip
        self.cached_model: (
            tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP, comfy.sd.VAE] | None
        ) = None


class ModelProcessor:
    @staticmethod
    def load_model(
        model: Model,
    ) -> tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP, comfy.sd.VAE]:
        if not model or not model.name:
            raise ValueError("Model name is required and cannot be empty")

        if model.cached_model is not None:
            return model.cached_model

        target_path = os.path.join(model.subfolder, model.name)
        ckpt_path = folder_paths.get_full_path("checkpoints", target_path)

        if not ckpt_path:
            raise FileNotFoundError(f"Checkpoint '{target_path}' not found.")

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        if not out or not out[0] or not out[1] or not out[2]:
            raise ValueError(f"Failed to load model from '{ckpt_path}'")

        output_model, clip, vae = (out[0], out[1], out[2])

        if model.clip_skip < 0:
            if model.clip_skip != -1:
                clip = clip.clone()
                clip.clip_layer(model.clip_skip)
                if hasattr(clip.cond_stage_model, "clip_layer"):
                    clip.cond_stage_model.set_clip_options({"layer": model.clip_skip})
        else:
            raise ValueError(f"Invalid clip_skip value: {model.clip_skip}")

        model.cached_model = (output_model, clip, vae)
        return (output_model, clip, vae)
