from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import csv
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

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/api/infer")
async def infer(
    image: UploadFile = File(...),
    bboxes: str = Form(...)
):
    import tempfile, json
    start = time.time()
    import mimetypes
    import cv2
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[-1]) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    # Parse bounding boxes from JSON string
    try:
        bboxes_list = json.loads(bboxes)
    except Exception:
        os.remove(tmp_path)
        return {"error": "Invalid bboxes format. Must be JSON list."}
    if len(bboxes_list) < 1:
        os.remove(tmp_path)
        return {"error": "At least one bounding box required."}

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

    # Create annotated frames that will be sent to the model
    annotated_frame_paths = []
    annotated_filenames = []
    base_name, _ = os.path.splitext(image.filename)
    for i, frame_path in enumerate(frame_paths):
        img = Image.open(frame_path).convert("RGB")
        # Highlight the target object only in the first frame
        if i == 0:
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = bboxes_list[0]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        annotated_name = f"{base_name}_annotated_{i}.png"
        annotated_path = os.path.join(IMAGES_DIR, annotated_name)
        img.save(annotated_path)
        annotated_frame_paths.append(annotated_path)
        annotated_filenames.append(annotated_name)

    content = []
    for idx, frame_path in enumerate(annotated_frame_paths):
        content.append({"type": "text", "text": f"<frame {idx}>: "})
        content.append({"type": "image", "image": frame_path})

    question = (
        "You are an expert visual tracker.\n"
        f"You are given {num_frames} frames from a video.\n"
        "In <frame 0>, the target object is highlighted by a red bounding box.\n"
        "For each frame, output the bounding box of the same object "
        "as normalized integer coordinates between 0 and 1000 in the following strict JSON format:\n"
        "[{\"frame\": t, \"bbox\": [x1, y1, x2, y2]}, ...]\n"
        "where x1, y1, x2, y2 are integers between 0 and 1000.\n"
        "Respond with JSON only and no additional text."
    ).format(last_frame=num_frames - 1)
    content.append({"type": "text", "text": question})

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

    # Try to parse the model output as JSON trajectory
    try:
        trajectory = json.loads(output_text)
    except Exception:
        trajectory = None

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
        "trajectory": trajectory,
        "latency": latency,
        "video_name": image.filename,
        "annotated_images": annotated_filenames,
    }
