# InkToText AI – Handwritten Notes Recognition System

Upload a photo or scan of handwritten English notes and get editable text. Uses OpenCV for line detection and Microsoft's pretrained **TrOCR (`microsoft/trocr-base-handwritten`)** for recognition. No MNIST.

## Features
Drag-and-drop upload with preview and validation · multi-line pages · per-line confidence and "check these lines" warnings · editable result beside the original image · Copy, Clear, Download TXT, Export PDF · sample note · processing time · temp files deleted after every request.

## Setup on Windows + VS Code
Requires Python 3.10–3.12 (tick "Add to PATH" in the installer) and ~3 GB free disk space.

1. Open the `InkToText-AI` folder in VS Code (File > Open Folder), then open a terminal (Terminal > New Terminal, PowerShell).
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
3. Install dependencies (CPU-only PyTorch is smaller and works everywhere):
   ```
   python -m pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
4. Run:
   ```
   python app.py
   ```
5. Open http://127.0.0.1:5000. In VS Code select the interpreter via Ctrl+Shift+P > "Python: Select Interpreter" > `venv`.

## How the model is downloaded and initialized
`app.py` starts a background thread that calls `ocr_engine.load_model()`. On the **first run** Hugging Face downloads the model (~1.3 GB) into `C:\Users\<you>\.cache\huggingface`; later runs load it from disk. The model is stored in a module-level singleton behind a lock, so it is loaded once and reused by every request (it uses the GPU automatically if CUDA PyTorch is installed). `GET /api/health` shows whether it is ready.

## Word detector (EasyOCR) – for mind maps, two-column and irregular pages
`pip install -r requirements.txt` also installs EasyOCR. Only its CRAFT *text detector* is used (about 80 MB, downloaded on first start). It finds every word/phrase anywhere on the page; boxes are grouped into rows and read by TrOCR. If EasyOCR is missing or fails, the app silently falls back to the line-projection method. Use "Page layout > Simple lines" in the UI to force the fallback on plain notebook pages.
If OpenCV errors appear after installing EasyOCR: `pip uninstall -y opencv-python opencv-python-headless` then `pip install opencv-python-headless`.

## Pipeline (line-projection fallback)
1. Grayscale, denoise, shadow flattening, CLAHE contrast, adaptive threshold, ruled-line removal (`preprocess`).
2. Line detection with a horizontal projection profile; very wide lines are split at ink gaps (`find_lines`, `split_wide`).
3. TrOCR beam search per line (batches of 8); confidence = exp(length-normalised log-probability).
4. Lines joined with newlines; larger vertical gaps become blank lines (paragraphs).

## API
`POST /api/recognize` (multipart field `image`) returns `{text, lines[{text, confidence}], avg_confidence, seconds}` or `{error}`. Max 10 MB. Extension, MIME type and real image content are validated; files are re-encoded and stored under a random UUID name, then deleted in a `finally` block.

## Limitations (please read)
- Accuracy depends heavily on handwriting style, ink contrast and photo quality. Expect mistakes; always proofread.
- English only (the selected model has no language option, so none is offered).
- Crossed-out words, margin notes, drawings, tables, columns, strongly rotated pages and very dense or overlapping lines may be misread or merged.
- Confidence scores are model estimates, not guarantees; a confident line can still be wrong.
- CPU speed: roughly 1–3 s per line.

## Troubleshooting
| Problem | Fix |
|---|---|
| First request is very slow | Model is downloading; wait for it to finish. |
| `ModuleNotFoundError` | Activate the venv, rerun `pip install -r requirements.txt`. |
| Download fails / SSL error | Check internet/proxy/firewall; delete a partial `.cache\huggingface` folder and retry. |
| PDF button error | Page needs internet for the jsPDF CDN script. |
| "No handwriting detected" | Use a brighter, sharper, straight-on photo with dark ink. |
| Lines merged or split wrongly | Crop to the text, photograph without shadows, keep lines horizontal. |
| Port 5000 busy | Change `port=5000` in `app.py`. |
| Out-of-memory | Close other apps or use smaller images. |
