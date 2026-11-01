"""Model loading survives checkpoints that lack weights the installed library expects (no downloads: tiny models)."""
import pytest
import torch

from unveil.config import GenerationConfig
from unveil.generation import SketchImageGenerator
from unveil.utils import empty_weights, load_without_empty_weights


def _tiny_controlnet(path, drop=None):
    diffusers = pytest.importorskip("diffusers")
    unet = diffusers.UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=1, sample_size=16, in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8,
    )
    diffusers.ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(8, 16)).save_pretrained(path)
    if drop:
        import safetensors.torch as st

        f = next(path.glob("*.safetensors"))
        state = st.load_file(f)
        del state[drop]
        st.save_file(state, f)
    return diffusers.ControlNetModel


def test_missing_checkpoint_weight_is_reloaded_instead_of_crashing(tmp_path, caplog):
    cls = _tiny_controlnet(tmp_path, drop="controlnet_cond_embedding.conv_in.weight")
    gen = SketchImageGenerator(GenerationConfig(), "cpu", torch.float32)
    model = gen._from_pretrained(cls, str(tmp_path))
    assert not empty_weights(model)
    model.to("cpu")  # the call that raised "Cannot copy out of meta tensor"
    assert "conv_in.weight" in caplog.text


def test_complete_checkpoint_loads_once(tmp_path):
    cls = _tiny_controlnet(tmp_path)
    calls = []

    def load(**extra):
        calls.append(extra)
        return cls.from_pretrained(str(tmp_path), **extra)

    assert not empty_weights(load_without_empty_weights(load, "tiny"))
    assert calls == [{}]  # no slow reload when nothing is missing
