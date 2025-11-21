import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
import torchvision
import tqdm
from dataset.dataset_utils import get_video_and_audio
from dataset.transforms import make_class_grid, quantize_offset
from omegaconf import OmegaConf
from scripts.train_utils import get_model, get_transforms, prepare_inputs
from torch.utils.data import DataLoader, Dataset, default_collate
from utils.utils import check_if_file_exists_else_download

memory = joblib.Memory(".cache", verbose=0)


def resize_collate_fn(batch):
    """Custom collate_fn that, filters out None items, resizes videos to the max
    height/width in a batch and then uses the default_collate function."""
    batch = [item for item in batch if item is not None]

    # If all items are corrupt, return None and let the caller handle it.
    if len(batch) == 0:
        return None

    # Resize video patches to the max height and width in the batch.
    heights = [item["video"].shape[-2] for item in batch]
    widths = [item["video"].shape[-1] for item in batch]
    max_h = max(heights)
    max_w = max(widths)
    resize_fn = torchvision.transforms.Resize((max_h, max_w))
    for item in batch:
        video = item["video"]

        num_patches, num_examples, c, h, w = video.shape
        if h < max_h or w < max_w:
            video = video.reshape(-1, c, h, w)
            video = resize_fn(video)
            video = video.reshape(num_patches, num_examples, c, max_h, max_w)

        item["video"] = video

    # Ensure all videos and audios have the same shape in the batch.
    v_shape = batch[0]["video"].shape
    a_shape = batch[0]["audio"].shape
    for item in batch:
        assert (
            item["video"].shape == v_shape
        ), f"Video shape mismatch: {item['video'].shape} vs {v_shape}"
        assert (
            item["audio"].shape == a_shape
        ), f"Audio shape mismatch: {item['audio'].shape} vs {a_shape}"

    return default_collate(batch)


@memory.cache()
def get_video_duration(video: Path) -> float:
    """Returns the duration of a video in seconds."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"ffprobe failed for {video}: {result.stderr}")
        return 0.0

    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


@memory.cache()
def get_media_info(video: Path):
    """Extract video frame rate, video resolution and audio sample rate."""

    # Get video frame rate and dimensions.
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(video),
    ]
    vinfo = json.loads(subprocess.check_output(cmd))["streams"][0]
    vfps = eval(vinfo["r_frame_rate"])
    width, height = vinfo["width"], vinfo["height"]

    # Get audio sample rate.
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "json",
            str(video),
        ]
        ainfo = json.loads(subprocess.check_output(cmd))
        afps = int(ainfo["streams"][0]["sample_rate"])
    except Exception:
        logging.warning(f"No audio stream found in {video.name}")
        afps = 0

    return {"video_fps": vfps, "audio_fps": afps, "width": width, "height": height}


@memory.cache()
def reencode_video(video: Path, vfps: int, afps: int, in_size: int) -> Path:
    info = get_media_info(video)
    H, W = info["height"], info["width"]

    needs_reencode = (
        abs(info["video_fps"] - vfps) > 1e-2
        or abs(info["audio_fps"] - afps) > 1
        or min(H, W) != in_size
    )

    if needs_reencode:
        logging.info(f"Reencoding {video.name}...")
        new_path = Path.cwd() / "reencoded" / video
        new_path.parent.mkdir(exist_ok=True, parents=True)
        cmd = "ffmpeg -hide_banner -loglevel panic"
        cmd += f" -y -i {str(video)}"
        cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
        cmd += f" -ar {afps}"
        cmd += f" {str(new_path)}"
        subprocess.call(cmd.split())

    return video


def patch_config(cfg):
    # the FE ckpts are already in the model ckpt
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    # old checkpoints have different names
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
        ".modules.feature_selector.", ".sync_model."
    )
    return cfg


class VideoDataset(Dataset):
    def __init__(
        self,
        videos: List[Path],
        cfg,
        vfps,
        afps,
        in_size,
        v_start_i_sec,
        offset_sec,
    ):
        self.videos = videos
        self.cfg = cfg
        self.vfps = vfps
        self.afps = afps
        self.in_size = in_size
        self.v_start_i_sec = v_start_i_sec
        self.offset_sec = offset_sec
        self.transforms = get_transforms(cfg, ["test"])["test"]
        self.grid = make_class_grid(
            leftmost_val=-cfg.data.max_off_sec,
            rightmost_val=cfg.data.max_off_sec,
            grid_size=cfg.model.params.transformer.params.off_head_cfg.params.out_features,
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]

        # Check and reencode if needed.
        video = reencode_video(video, self.vfps, self.afps, self.in_size)

        # Load video and audio.
        rgb, audio, meta = torchvision.io.read_video(
            filename=str(video),
            start_pts=0,
            end_pts=None,
            pts_unit="sec",
            output_format="TCHW",
        )
        # TODO Ensure audio is in expected range [-1.0, 1.0], important or not?
        # print(audio.max(), audio.min(), audio.dtype, audio.shape)
        audio = audio.float().mean(dim=0)
        meta = {
            "video": {"fps": [meta["video_fps"]]},
            "audio": {"framerate": [meta["audio_fps"]]},
        }
        targets = {
            "v_start_i_sec": self.v_start_i_sec,
            "offset_sec": self.offset_sec,
        }

        # Collect features and targets.
        item = {
            "video": rgb,
            "audio": audio,
            "meta": meta,
            "path": str(video),
            "split": "test",
            "grid": self.grid,
            "targets": targets,
        }

        # Apply transforms.
        try:
            item = self.transforms(item)
        except AssertionError as e:
            logging.warning(f"Skipping {video.name} due to {e}")
            return None

        # Make sure tensors are contiguous.
        if isinstance(item["video"], torch.Tensor):
            item["video"] = item["video"].cpu().float().contiguous().clone()
        if isinstance(item["audio"], torch.Tensor):
            item["audio"] = item["audio"].cpu().float().contiguous().clone()

        return item


class VideoSyncInferenceModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.results = []

    def forward(self, vid, aud):
        _, logits = self.model(vid, aud)
        return logits

    def predict_step(self, batch, batch_idx):
        aud, vid, _ = prepare_inputs(batch, self.device, get_targets=False)

        with torch.no_grad():
            with torch.autocast("cuda", enabled=self.cfg.training.use_half_precision):
                logits = self(vid, aud)

        offset_probs = torch.softmax(logits, dim=-1)
        pred_indices = torch.argmax(offset_probs, dim=-1)
        scores = offset_probs[range(len(offset_probs)), pred_indices]
        for path, label, grid, score in zip(
            batch["path"], pred_indices, batch["grid"], scores
        ):
            offset = grid[label]

            *_, system, dataset, video = Path(path).parts
            row = {
                "video": video,
                "dataset": dataset,
                "system": system,
                "score": offset.item(),
                "probability": score.item(),
            }
            logging.info(row)
            self.results.append(row)


def load_model(ckpt_path, cfg, device):
    _, model = get_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    cfg = patch_config(cfg)
    return cfg


def main(args):
    vfps = 25
    afps = 16000
    in_size = 256
    cfg_path = f"./logs/sync_models/{args.exp_name}/cfg-{args.exp_name}.yaml"
    ckpt_path = f"./logs/sync_models/{args.exp_name}/{args.exp_name}.pt"
    device = torch.device(args.device)

    # Download if needed
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    # Load config.
    cfg = load_config(cfg_path)

    # Load model.
    model = load_model(ckpt_path, cfg, device)

    # Find videos.
    indir = Path(args.video_dir)
    videos = sorted(list(indir.rglob("*.mp4")))
    logging.info(f"Found {len(videos)} videos")

    # Skip any video that is too short.
    min_duration = 5.0
    with ThreadPoolExecutor(max_workers=8) as executor:
        durations = executor.map(get_video_duration, videos)
        videos = [
            video
            for video, duration in zip(videos, durations)
            if duration >= min_duration
        ]
    logging.info(f"{len(videos)} videos kept after filtering short ones.")

    # Create dataset and dataloader.
    dataset = VideoDataset(
        videos=videos,
        cfg=cfg,
        vfps=vfps,
        afps=afps,
        in_size=in_size,
        v_start_i_sec=args.v_start_i_sec,
        offset_sec=args.offset_sec,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=resize_collate_fn,
    )

    # Run inference
    inference_module = VideoSyncInferenceModule(model, cfg)
    trainer = pl.Trainer(accelerator="auto", logger=False)
    trainer.predict(inference_module, dataloader)

    # Save results
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(inference_module.results)
    df.to_csv(outdir / "results.csv", index=False)
    logging.info(f'Results saved to {outdir / "results.csv"}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name", required=True, help="In a format: xx-xx-xxTxx-xx-xx"
    )
    parser.add_argument(
        "--video-dir", required=True, help="A path to directory of .mp4 videos"
    )
    parser.add_argument(
        "--output-dir", required=True, help="A path to directory to save results"
    )
    parser.add_argument("--offset-sec", type=float, default=0.0)
    parser.add_argument("--v-start-i-sec", type=float, default=0.0)
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of dataloader workers"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")
    main(args)
