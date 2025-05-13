#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Grounding-DINO on an MP4 to detect sports balls and produce an annotated MP4"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input video (mp4)")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to output video (mp4)")
    parser.add_argument("--box-threshold", type=float, default=0.25,
                        help="Bounding box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.3,
                        help="Text (label) confidence threshold")
    parser.add_argument("--prompt", type=str, default="sports ball.",
                        help="Text prompt for detection (lowercase, ending with a period)")
    parser.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-base",
                        help="🤗 Hub model identifier for Grounding-DINO")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.input}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Set up output video writer
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Then open your writer on the resolved absolute path:
    out = cv2.VideoWriter(
        str(output_path.resolve()),
        fourcc,
        fps,
        (width, height)
    )

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Convert BGR -> RGB PIL image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Inference
        inputs = processor(images=img_pil, text=args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        # Post-process
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(height, width)]
        )
        detection = results[0]
        if "boxes" in detection and detection["boxes"].numel() > 0:
            boxes = detection["boxes"].cpu().numpy()
            for (x1, y1, x2, y2) in boxes.astype(int):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            args.prompt.rstrip("."),
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA)

        # Write annotated frame
        out.write(frame)

        # (optional) print progress
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")

    cap.release()
    
    # After you release it, print the absolute path:
    out.release()
    print(f"Done! Wrote annotated video to {output_path.resolve()}")

if __name__ == "__main__":
    main()
