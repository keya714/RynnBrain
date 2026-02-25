from fastapi import FastAPI, Request
import time
import os
from PIL import Image
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor

app = FastAPI()

@app.post("/api/infer")
async def infer(request: Request):
    start = time.time()
    data = await request.json()
    image_path = data.get("image_path")
    bboxes = data.get("bboxes", [])
    # Validate input
    if not image_path or not os.path.exists(image_path):
        return {"error": "Image path not found"}
    if len(bboxes) < 2:
        return {"error": "At least two bounding boxes required"}

    w, h = Image.open(image_path).size
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

    bbox_str_0 = convert_bbox(bboxes[0], w, h)
    bbox_str_1 = convert_bbox(bboxes[1], w, h)
    content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": f"What is the distance between <object>; {bbox_str_0} </object> and <object>; {bbox_str_1} </object>?"}
    ]
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
    return {"distance": output_text, "latency": latency}

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class BoundingBoxSelector:
    def __init__(self, image_path):
        self.image = np.array(Image.open(image_path))
        self.bboxes = []
        self.current_bbox = None
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.rect = None
        plt.title("Draw bounding boxes by clicking and dragging. Close window when done.")
        plt.show()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.current_bbox = [event.xdata, event.ydata, event.xdata, event.ydata]
        self.rect = plt.Rectangle((event.xdata, event.ydata), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
        self.ax.add_patch(self.rect)
        self.fig.canvas.draw()

    def on_motion(self, event):
        if self.current_bbox is None or event.inaxes != self.ax:
            return
        self.current_bbox[2] = event.xdata
        self.current_bbox[3] = event.ydata
        x0, y0, x1, y1 = self.current_bbox
        self.rect.set_width(x1 - x0)
        self.rect.set_height(y1 - y0)
        self.fig.canvas.draw()

    def on_release(self, event):
        if self.current_bbox is None or event.inaxes != self.ax:
            return
        self.current_bbox[2] = event.xdata
        self.current_bbox[3] = event.ydata
        x0, y0, x1, y1 = self.current_bbox
        bbox = [int(x0), int(y0), int(x1), int(y1)]
        self.bboxes.append(bbox)
        self.current_bbox = None
        self.rect = None
        self.fig.canvas.draw()

def calculate_distance(bbox1, bbox2):
    # Closest point between two rectangles
    x1 = np.clip(bbox2[0], bbox1[0], bbox1[2])
    y1 = np.clip(bbox2[1], bbox1[1], bbox1[3])
    x2 = np.clip(bbox1[0], bbox2[0], bbox2[2])
    y2 = np.clip(bbox1[1], bbox2[1], bbox2[3])
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist

if __name__ == "__main__":
    image_path = "image.png"  # Replace with your image path
    selector = BoundingBoxSelector(image_path)
    print("Bounding boxes:", selector.bboxes)
    if len(selector.bboxes) >= 2:
        # Prepare bounding box data for RynnBrain
        from PIL import Image
        w, h = Image.open(image_path).size
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

        bbox_str_0 = convert_bbox(selector.bboxes[0], w, h)
        bbox_str_1 = convert_bbox(selector.bboxes[1], w, h)

        # Prepare message for RynnBrain
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"What is the distance between <object>; {bbox_str_0} </object> and <object>; {bbox_str_1} </object>?"}
        ]
        messages = [
            {"role": "user", "content": content}
        ]

        # Load RynnBrain model and processor
        from transformers import AutoModelForImageTextToText, AutoProcessor
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
        print("RynnBrain model output (distance):", output_text)
    else:
        print("Please select at least two bounding boxes.")