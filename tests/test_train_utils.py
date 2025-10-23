import pytest
import torch

from scripts.train_utils import prepare_inputs


@pytest.mark.parametrize("get_targets", [True, False])
def test_prepare_inputs(get_targets):

    video_shape = (2, 3, 224, 224)
    audio_shape = (2, 16000)
    batch = {
        "video": torch.randn(*video_shape),
        "audio": torch.randn(*audio_shape),
        "targets": {
            "offset_target": torch.tensor([0, 1]),
        },
    }

    device = "cpu"
    phase = "train"
    audio, video, targets = prepare_inputs(batch, device, phase, get_targets)

    assert audio.shape == audio_shape
    assert video.shape == video_shape

    if get_targets:
        assert targets is not None
        assert targets["offset_target"].shape == (2,)
