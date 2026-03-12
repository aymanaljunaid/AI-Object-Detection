# ⚡ AI Object Detection

Real-time object detection running **entirely in your browser** — no server, no cloud, no API keys. Uses a YOLOv8n model via [ONNX Runtime Web](https://onnxruntime.ai/) with WebGPU acceleration and automatic WASM fallback.

![Dark Mode UI](https://img.shields.io/badge/theme-dark_mode-0a0f1a?style=flat-square)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8n-10b981?style=flat-square)
![WebGPU](https://img.shields.io/badge/backend-WebGPU-10b981?style=flat-square)

---

## ✨ Features

- 🎯 **Real-time inference** from your webcam — detects objects frame-by-frame
- ⚡ **WebGPU acceleration** with automatic WASM fallback for older browsers
- 🏷️ **80 COCO classes** — people, vehicles, animals, food, furniture, electronics & more
- 🌓 **Dark / Light mode** with one-click toggle
- 📊 **Live diagnostics panel** — model status, FPS, inference time, pipeline health
- 🎛️ **Adjustable thresholds** — tune confidence & NMS (IoU) in real time
- 📥 **Export detections** as JSON or CSV for analysis
- 🔒 **100% client-side** — your camera feed never leaves your device

---

## 🚀 Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) 18+ (LTS recommended)
- A modern browser with WebGPU support (Chrome 113+, Edge 113+) — WASM works everywhere

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/aymanaljunaid/ai-object-detection.git
cd ai-object-detection

# 2. Install dependencies
npm install

# 3. Download the YOLOv8n ONNX model (~213 MB)
npm run download-model

# 4. Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser, click **Start**, and allow camera access.

> **💡 Tip:** The model is ~213 MB and may take 10–20 seconds to load on first run. You'll see a loading progress bar.

---

## 🏷️ What Can It Detect?

The model can detect **80 object categories** from the [COCO dataset](https://cocodataset.org/):

<details>
<summary><strong>View all 80 classes</strong></summary>

| # | Category | # | Category | # | Category | # | Category |
|---|----------|---|----------|---|----------|---|----------|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
| 10 | fire hydrant | 30 | skis | 50 | broccoli | 70 | toaster |
| 11 | stop sign | 31 | snowboard | 51 | carrot | 71 | sink |
| 12 | parking meter | 32 | sports ball | 52 | hot dog | 72 | refrigerator |
| 13 | bench | 33 | kite | 53 | pizza | 73 | book |
| 14 | bird | 34 | baseball bat | 54 | donut | 74 | clock |
| 15 | cat | 35 | baseball glove | 55 | cake | 75 | vase |
| 16 | dog | 36 | skateboard | 56 | chair | 76 | scissors |
| 17 | horse | 37 | surfboard | 57 | couch | 77 | teddy bear |
| 18 | sheep | 38 | tennis racket | 58 | potted plant | 78 | hair drier |
| 19 | cow | 39 | bottle | 59 | bed | 79 | toothbrush |

</details>

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv8n (nano) |
| **Export format** | ONNX (end2end, NMS built-in) |
| **Input** | `[1, 3, 640, 640]` — single RGB image, 640×640 |
| **Output** | `[1, 300, 6]` — up to 300 detections |
| **Per-detection format** | `[x1, y1, x2, y2, confidence, class_id]` |
| **File size** | ~213 MB |
| **Classes** | 80 (COCO) |
| **Source** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |

### Using a Different Model

You can swap the model for any YOLOv8 ONNX export. Place it in `public/models/` and update the model path in `src/app/page.tsx`:

```typescript
const MODEL_PATH = '/models/your-model.onnx';
```

The app auto-detects two output formats:

| Format | Output shape | Example |
|--------|-------------|---------|
| **End2End** (recommended) | `[1, N, 6]` | YOLOv8 with `end2end=True` |
| **Raw YOLO** | `[1, 84, 8400]` | Standard YOLOv8 export |

#### Exporting your own YOLOv8 model

```bash
pip install ultralytics

# End2End export (recommended — NMS built-in, smaller output)
yolo export model=yolov8n.pt format=onnx simplify=True

# Or use the Python API
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', simplify=True)
```

> **📝 Model size variants:** `yolov8n` (nano, ~12 MB raw / ~213 MB end2end) is the fastest. Larger variants (`yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`) are more accurate but slower in-browser.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | [Next.js](https://nextjs.org/) 16 |
| Inference | [ONNX Runtime Web](https://onnxruntime.ai/) |
| Acceleration | WebGPU (fallback: WASM) |
| Model | [YOLOv8n](https://github.com/ultralytics/ultralytics) |
| Styling | [Tailwind CSS](https://tailwindcss.com/) v4 + [shadcn/ui](https://ui.shadcn.com/) |
| Theming | [next-themes](https://github.com/pacocoursey/next-themes) |
| Language | TypeScript |

---

## 📁 Project Structure

```
Detector/
├── public/
│   └── models/
│       └── yolov8n.onnx          # ONNX model file
├── src/
│   ├── app/
│   │   ├── page.tsx              # Main UI page
│   │   ├── layout.tsx            # Root layout + ThemeProvider
│   │   └── globals.css           # Theme colors & animations
│   ├── components/
│   │   ├── DetectionCanvas.tsx    # Draws bounding boxes on camera
│   │   ├── ThemeProvider.tsx      # Dark/light mode provider
│   │   ├── ThemeToggle.tsx        # Theme toggle button
│   │   └── ui/                   # shadcn/ui components
│   ├── hooks/
│   │   ├── useCamera.ts          # Camera access & frame capture
│   │   └── useDetection.ts       # Detection pipeline orchestration
│   └── lib/
│       └── detection/
│           ├── detector.ts       # ONNX model loading & inference
│           ├── postprocess.ts    # NMS & box filtering
│           ├── preprocess.ts     # Image resize & tensor creation
│           ├── eventLogger.ts    # Detection event logging
│           └── types.ts          # Type definitions & COCO classes
├── package.json
├── next.config.ts
├── tailwind.config.ts
└── tsconfig.json
```

---

## ⚙️ Configuration

### Detection Settings (UI)

| Setting | Default | Description |
|---------|---------|-------------|
| Confidence Threshold | 50% | Minimum confidence to show a detection |
| IoU Threshold (NMS) | 45% | Overlap threshold for non-maximum suppression |
| Show Labels | On | Display class names on bounding boxes |
| Show Confidence | On | Display confidence percentages |
| Show FPS | On | Display FPS overlay on video |

### Browser Compatibility

| Browser | WebGPU | WASM Fallback |
|---------|--------|---------------|
| Chrome 113+ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ |
| Firefox | ❌ | ✅ |
| Safari 18+ | ✅ | ✅ |

> WebGPU provides significantly better performance. If your browser doesn't support WebGPU, the app automatically falls back to WASM (slower but universal).

---

## 📤 Exporting Data

The app supports exporting detection events in two formats:

- **JSON** — Full event objects with timestamps, coordinates, and confidence scores
- **CSV** — Spreadsheet-friendly format for analysis in Excel or Google Sheets

Click the **JSON** or **CSV** button in the Detection Settings panel to download.

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Camera not started" | Allow camera permissions in your browser |
| Model takes long to load | Normal — the model is ~213 MB. Wait for the progress bar |
| Low FPS | Use Chrome/Edge for WebGPU acceleration |
| No detections appear | Lower the confidence threshold in settings |
| "2 Issues" badge (dev only) | Harmless ONNX Runtime warnings — click ✕ to dismiss |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).