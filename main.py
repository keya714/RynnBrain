from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import time
import os
import csv
import re
from PIL import Image, ImageDraw
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor

IMAGES_DIR = "images"
CSV_PATH = "results.csv"

os.makedirs(IMAGES_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "annotated_images", "response", "latency"])

MODEL_PATH = "Alibaba-DAMO-Academy/RynnBrain-8B"
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.post("/api/infer")
async def infer(
    image: UploadFile = File(...),
    bboxes: str = Form("[]")
):
    import tempfile, json
    start = time.time()
    import mimetypes
    import cv2
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[-1]) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    # Determine file type by extension (robust for temp files)
    ext = os.path.splitext(image.filename)[-1].lower()
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    is_video = ext in video_exts
    is_image = ext in image_exts

    frame_paths = []
    if is_video:
        # Extract frames from video
        cap = cv2.VideoCapture(tmp_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_samples = min(n_frames, 8)
        idxs = np.linspace(0, n_frames - 1, num_samples).round().astype(int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            frame_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img.save(frame_tmp.name)
            frame_paths.append(frame_tmp.name)
        cap.release()
        if not frame_paths:
            os.remove(tmp_path)
            return {"error": "Could not extract frames from video."}
        w, h = Image.open(frame_paths[0]).size
    elif is_image:
        frame_paths = [tmp_path]
        w, h = Image.open(tmp_path).size
    else:
        os.remove(tmp_path)
        return {"error": "Unsupported file type. Please upload an image or video."}

    num_frames = len(frame_paths)

    # Create frames that will be sent to the model (no user-provided bbox guidance)
    annotated_frame_paths = []
    annotated_filenames = []
    base_name, _ = os.path.splitext(image.filename)
    for i, frame_path in enumerate(frame_paths):
        img = Image.open(frame_path).convert("RGB")
        annotated_name = f"{base_name}_annotated_{i}.png"
        annotated_path = os.path.join(IMAGES_DIR, annotated_name)
        img.save(annotated_path)
        annotated_frame_paths.append(annotated_path)
        annotated_filenames.append(annotated_name)

    content = []
    for idx, frame_path in enumerate(annotated_frame_paths):
        content.append({"type": "text", "text": f"<frame {idx}>: "})
        content.append({"type": "image", "image": frame_path})

    instruction = (
        "You are an expert trajectory analyst. "
        "You are given several frames from a video containing a tank. "
        "Your task is to predict the movement trajectory of the tank."
    )
    format_prompt = (
        "First predict the frame containing the trajectory start point, "
        "then output up to 10 key trajectory points as a list of tuples in the format: "
        ": ...; (x1, y1), (x2, y2), ....  "
        "All coordinates must be normalized between 0 and 1000."
    )
    content.append({"type": "text", "text": f"{instruction}\n{format_prompt}"})

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse trajectory points from the model's textual output, e.g.:
    # "The start trajectory is at ; (498, 487), (536, 491), ..."
    trajectory_points_norm = []
    for match in re.finditer(r"\((\d+)\s*,\s*(\d+)\)", output_text):
        x = int(match.group(1))
        y = int(match.group(2))
        trajectory_points_norm.append([x, y])

    trajectory_points_pixel = []
    if trajectory_points_norm:
        for x_norm, y_norm in trajectory_points_norm:
            x_px = int(round(x_norm / 1000 * (w - 1)))
            y_px = int(round(y_norm / 1000 * (h - 1)))
            trajectory_points_pixel.append([x_px, y_px])

    # Draw trajectory on all frames and save as new images
    trajectory_image_filenames = []
    if trajectory_points_pixel:
        for i, frame_path in enumerate(annotated_frame_paths):
            img = Image.open(frame_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            # Draw polyline for trajectory
            if len(trajectory_points_pixel) >= 2:
                draw.line(trajectory_points_pixel, fill="red", width=3)
            # Draw small circles at each key point
            r = 5
            for x_px, y_px in trajectory_points_pixel:
                draw.ellipse(
                    [x_px - r, y_px - r, x_px + r, y_px + r],
                    outline="yellow",
                    width=2,
                )
            traj_name = f"{base_name}_traj_{i}.png"
            traj_path = os.path.join(IMAGES_DIR, traj_name)
            img.save(traj_path)
            trajectory_image_filenames.append(traj_name)

    latency = time.time() - start

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            image.filename,
            "|".join(annotated_filenames),
            output_text,
            latency
        ])

    os.remove(tmp_path)
    return {
        "trajectory_raw": output_text,
        "trajectory_points_norm": trajectory_points_norm,
        "trajectory_points_pixel": trajectory_points_pixel,
        "trajectory_images": trajectory_image_filenames,
        "latency": latency,
        "video_name": image.filename,
        "annotated_images": annotated_filenames,
    }
