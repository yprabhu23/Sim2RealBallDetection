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
        description="Append sports‐ball frames to a COCO dataset from an MP4"
    )
    p.add_argument("-i","--input",     required=True, help="Input MP4")
    p.add_argument("-o","--output-dir",default="labeled_dataset",
                   help="Dataset root (will contain `images/` + `annotations.json`)")
    p.add_argument("--box-threshold", type=float, default=0.25)
    p.add_argument("--text-threshold",type=float, default=0.3)
    p.add_argument("--prompt",       default="sports ball.",
                   help="Lowercase, ending in a period")
    p.add_argument("--model-id",     default="IDEA-Research/grounding-dino-base")
    return p.parse_args()

def main():
    args = parse_args()

    # Prepare output dirs & annotation file
    out_dir    = Path(args.output_dir)
    images_dir = out_dir/"images"
    ann_path   = out_dir/"annotations.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    # If there's an existing COCO, load it; otherwise initialize
    if ann_path.exists():
        coco = json.loads(ann_path.read_text())
        existing_files = {img["file_name"] for img in coco["images"]}
        max_image_id   = max(img["id"]          for img in coco["images"])
        max_ann_id     = max(a["id"]            for a   in coco["annotations"])
    else:
        coco = {
            "images":      [],
            "annotations": [],
            "categories":  [{"id":1,"name":"sports_ball"}]
        }
        existing_files = set()
        max_image_id, max_ann_id = 0, 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("→ Using device:", device)

    # Load model
    proc  = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.input}")

    video_stem = Path(args.input).stem
    frame_idx  = 0
    next_image_id = max_image_id + 1
    next_ann_id   = max_ann_id   + 1
    last_frame = None
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1) skip exact-duplicate frame
        if last_frame is not None and np.array_equal(frame, last_frame):
            continue
        last_frame = frame.copy()

        # 2) run Grounding-DINO
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
            target_sizes=[frame.shape[:2]]
        )[0]

        # 3) skip if no detection
        if det.get("boxes") is None or det["boxes"].numel() == 0:
            continue

        # 4) determine filename & skip if it's already in your dataset
        fname = f"{video_stem}_{frame_idx:06d}.png"
        if fname in existing_files:
            continue

        # 5) save image
        cv2.imwrite(str(images_dir/fname), frame)

        # 6) append new image entry
        h, w = frame.shape[:2]
        coco["images"].append({
            "id":        next_image_id,
            "file_name": fname,
            "width":     w,
            "height":    h
        })
        existing_files.add(fname)

        # 7) append all boxes for this image
        for box in det["boxes"].cpu().numpy():
            x1,y1,x2,y2 = box.tolist()
            bw, bh = x2-x1, y2-y1
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

        # Progress
        if frame_idx % 200 == 0:
            elapsed = time.time() - t0
            saved = next_image_id - (max_image_id+1)
            print(f" → scanned {frame_idx} frames, saved {saved} images "
                  f"({frame_idx/elapsed:.1f} fps)")

    cap.release()

    # Write updated COCO JSON back
    ann_path.write_text(json.dumps(coco, indent=2))
    total_saved = next_image_id - (max_image_id+1)
    total_boxes = len(coco["annotations"]) - max_ann_id
    print(f"✅ Done: appended {total_saved} images + {total_boxes} boxes → {out_dir.resolve()}")

if __name__ == "__main__":
    main()
