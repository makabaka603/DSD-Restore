import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from models import DSDRestoreV1
from utils.runtime import configure_cuda, ensure_dir, get_amp_settings, get_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})
    configure_cuda(device, runtime_cfg.get("cudnn_benchmark", True))
    amp_enabled, amp_dtype = get_amp_settings(
        device,
        runtime_cfg.get("amp", True),
        runtime_cfg.get("amp_dtype", "bfloat16"),
    )
    model = DSDRestoreV1(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    input_dir = Path(args.input)
    output_dir = ensure_dir(args.output)
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    for path in sorted([p for p in input_dir.iterdir() if p.suffix.lower() in suffixes]):
        image = Image.open(path).convert("RGB")
        tensor = TF.to_tensor(image).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            restored = model(tensor)["restored"].squeeze(0).cpu()
        TF.to_pil_image(restored.clamp(0, 1)).save(output_dir / path.name)
        print(f"saved {output_dir / path.name}")


if __name__ == "__main__":
    main()
