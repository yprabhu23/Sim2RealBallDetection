#!/usr/bin/env python3
import argparse, time, json
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def parse_args():
    p = argparse.ArgumentParser(
        description="Append sports-ball images from a folder into a COCO dataset"
    )
    p.add_argument("-i","--input-dir", required=True,
                   help="Folder of input images (png/jpg/etc.)")
    p.add_argument("-o","--output-dir", default="labeled_synthetic_dataset",
                   help="Root output dir (will contain `images/` + `annotations.json`)")
    p.add_argument("--box-threshold",  type=float, default=0.25)
    p.add_argument("--text-threshold", type=float, default=0.3)
    p.add_argument("--prompt", default="sports ball.",
                   help="Lowercase, ending in a period")
    p.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    return p.parse_args()

def main():
    args = parse_args()

    #── prepare output dirs + load or init COCO JSON ─────────────────────────────
    out_dir    = Path(args.output_dir)
    images_dir = out_dir / "images"
    ann_path   = out_dir / "annotations.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    if ann_path.exists():
        coco = json.loads(ann_path.read_text())
        existing_files = {img["file_name"] for img in coco["images"]}
        max_img_id = max(img["id"] for img in coco["images"])
        max_ann_id = max(a["id"]   for a   in coco["annotations"])
    else:
        coco = {
            "images":      [],
            "annotations": [],
            "categories":  [{"id":1,"name":"sports_ball"}]
        }
        existing_files = set()
        max_img_id, max_ann_id = 0, 0

    next_image_id = max_img_id + 1
    next_ann_id   = max_ann_id  + 1

    #── load DINO ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("→ Using device:", device)
    proc  = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    #── iterate images ──────────────────────────────────────────────────────────
    folder = Path(args.input_dir)
    if not folder.is_dir():
        raise RuntimeError(f"{folder} is not a valid directory")

    last_frame = None
    frame_idx = 0
    t0 = time.time()

    # only take common image extensions:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    total_saved = 0  # total number of saved images
    for img_fp in sorted(folder.iterdir()):
        if img_fp.suffix.lower() not in exts:
            continue
        frame_idx += 1

        frame = cv2.imread(str(img_fp))
        if frame is None:
            continue

        if last_frame is not None and np.array_equal(frame, last_frame):
            continue
        last_frame = frame.copy()

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs  = proc(images=img_pil, text=args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            if device=="cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs)
            else:
                out = model(**inputs)

        det = proc.post_process_grounded_object_detection(
            out, inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[frame.shape[:2]]  # (H, W)
        )[0]

        if det.get("boxes") is None or det["boxes"].numel() == 0:
            continue

        stem = folder.stem
        fname = f"{stem}_{img_fp.stem}.png"
        if fname in existing_files:
            continue

        cv2.imwrite(str(images_dir/fname), frame)

        h, w = frame.shape[:2]
        coco["images"].append({
            "id":        next_image_id,
            "file_name": fname,
            "width":     w,
            "height":    h
        })
        existing_files.add(fname)

        for box in det["boxes"].cpu().numpy():
            x1, y1, x2, y2 = box.tolist()
            bw, bh = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id":           next_ann_id,
                "image_id":     next_image_id,
                "category_id":  1,
                "bbox":         [x1, y1, bw, bh],
                "area":         bw * bh,
                "iscrowd":      0
            })
            next_ann_id += 1

        next_image_id += 1
        total_saved += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"→ scanned {frame_idx} imgs, saved {total_saved} "
                  f"({frame_idx/(elapsed):.1f} fps)")

        if total_saved >= 10000:
            print("🛑 Reached limit of 10,000 saved images. Stopping...")
            break

    #── write back JSON ─────────────────────────────────────────────────────────
    ann_path.write_text(json.dumps(coco, indent=2))
    total_saved = next_image_id - (max_img_id + 1)
    total_boxes = len(coco["annotations"]) - max_ann_id
    print(f"✅ Done: appended {total_saved} images + {total_boxes} boxes to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
