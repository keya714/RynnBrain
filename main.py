from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import time
import os
from PIL import Image
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ay31j3vw8jhy14-8000.proxy.runpod.net",
        "http://ay31j3vw8jhy14-8000.proxy.runpod.net"
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
    if len(bboxes_list) < 2:
        os.remove(tmp_path)
        return {"error": "At least two bounding boxes required"}

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
    def convert_bbox(bbox, w, h):
        x1, y1, x2, y2 = bbox
        bbox_norm = [
            int(round(x1 / (w-1) * 1000)),
            int(round(y1 / (h-1) * 1000)),
            int(round(x2 / (w-1) * 1000)),
            int(round(y2 / (h-1) * 1000)),
        ]
        bbox_norm = [max(0, min(1000, v)) for v in bbox_norm]
        return f"({bbox_norm[0]}, {bbox_norm[1]}), ({bbox_norm[2]}, {bbox_norm[3]})"

    content = []
    for idx, frame_path in enumerate(frame_paths):
        content.append({"type": "text", "text": f"<frame {idx}>: "})
        content.append({"type": "image", "image": frame_path})

    bbox_strs = []
    for i, bbox in enumerate(bboxes_list):
        bbox_str = convert_bbox(bbox, w, h)
        bbox_strs.append(f"<object> <frame{i}>; {bbox_str} </object>")

    question = f"What is the distance between {bbox_strs[0]} and {bbox_strs[1]}?"
    content.append({"type": "text", "text": question})

    messages = [
        {"role": "user", "content": content}
    ]

    model_path = "Alibaba-DAMO-Academy/RynnBrain-8B"
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

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
    latency = time.time() - start
    os.remove(tmp_path)
    return {"distance": output_text, "latency": latency}
