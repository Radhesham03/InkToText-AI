"""InkToText AI - Flask backend."""
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont

import ocr_engine

BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "static" / "uploads"
SAMPLE_DIR = BASE / "static" / "sample"
ALLOWED = {"jpg", "jpeg", "png", "bmp", "gif", "webp", "tif", "tiff"}
MIMES = {"image/jpeg", "image/png", "image/bmp", "image/gif", "image/webp", "image/tiff"}
MAX_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def make_sample():
    """Create a test note with a handwriting-style font if one exists on this machine."""
    SAMPLE_DIR.mkdir(exist_ok=True)
    out = SAMPLE_DIR / "sample_note.png"
    if out.exists():
        return out
    fonts = ["C:/Windows/Fonts/segoesc.ttf", "C:/Windows/Fonts/Inkfree.ttf",
             "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"]
    font = next((ImageFont.truetype(f, 54) for f in fonts if os.path.exists(f)), ImageFont.load_default())
    img = Image.new("RGB", (1300, 520), "#fdfcf7")
    d = ImageDraw.Draw(img)
    lines = ["Meeting notes - 12 March", "Buy 3 notebooks and 12 pens", "Call Priya about the project",
             "Deadline is on Friday at 5 pm"]
    for i, t in enumerate(lines):
        y = 40 + i * 115
        d.line([(40, y + 85), (1260, y + 85)], fill="#c9d6f2", width=2)
        d.text((60, y), t, fill="#1c2a6b", font=font)
    img.save(out)
    return out


@app.route("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)


@app.route("/api/health")
def health():
    return jsonify(model_ready=ocr_engine.model_ready(), model=ocr_engine.MODEL_NAME,
                   word_detector=ocr_engine.detector_ready())


@app.route("/api/sample")
def sample():
    p = make_sample()
    return send_from_directory(p.parent, p.name)


def validate(file):
    if not file or not file.filename:
        return "No file was selected."
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED:
        return "Unsupported file type. Please upload JPG, PNG, BMP, GIF, WEBP or TIFF."
    if file.mimetype not in MIMES:
        return "The file does not look like an image (unexpected MIME type)."
    return None


@app.route("/api/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")
    err = validate(file)
    if err:
        return jsonify(error=err), 400
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}.png"  # safe server-chosen name; original name never used
    try:
        try:
            img = Image.open(file.stream)
            img.verify()
            file.stream.seek(0)
            img = Image.open(file.stream)
            img.seek(0)
            img.convert("RGB").save(path, "PNG")  # re-encode: uploaded bytes are never kept or executed
        except Exception:
            return jsonify(error="The file is corrupted or not a valid image."), 400
        start = time.time()
        pref = "lines" if request.form.get("method") == "lines" else "auto"
        result = ocr_engine.recognize_image(str(path), pref)
        result["seconds"] = round(time.time() - start, 2)
        if not result["text"]:
            result["warning"] = "No handwriting was detected. Try a sharper, well-lit photo with dark ink."
        return jsonify(result)
    except Exception as e:  # noqa
        app.logger.exception("Recognition failed")
        return jsonify(error=f"Recognition failed: {e}"), 500
    finally:
        path.unlink(missing_ok=True)  # never keep private notes


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File is too large. Maximum size is {MAX_MB} MB."), 413


if __name__ == "__main__":
    threading.Thread(target=ocr_engine.warm_up, daemon=True).start()  # warm up model at startup
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
