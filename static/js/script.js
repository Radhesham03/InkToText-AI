const $ = id => document.getElementById(id);
const OK_EXT = ["jpg","jpeg","png","bmp","gif","webp","tif","tiff"];
let currentFile = null, previewURL = null;
const show = (el, on) => el.classList.toggle("d-none", !on);
function showError(msg){ $("errorBox").textContent = msg; show($("errorBox"), !!msg); }

function validate(file){
  const ext = file.name.split(".").pop().toLowerCase();
  if (!OK_EXT.includes(ext)) return "Unsupported file type. Please upload JPG, PNG, BMP, GIF, WEBP or TIFF.";
  if (file.size > MAX_MB * 1024 * 1024) return `File is too large (${(file.size/1048576).toFixed(1)} MB). Maximum is ${MAX_MB} MB.`;
  if (file.size === 0) return "The file is empty.";
  return null;
}
function setFile(file){
  showError("");
  const err = validate(file);
  if (err){ showError(err); return; }
  currentFile = file;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  $("previewImg").src = previewURL;
  $("fileName").textContent = file.name;
  $("fileMeta").textContent = `${(file.size/1024).toFixed(0)} KB`;
  show($("previewBox"), true); show($("results"), false);
}
function removeFile(){ currentFile = null; $("fileInput").value = ""; show($("previewBox"), false); showError(""); }

const dz = $("dropzone");
dz.addEventListener("click", () => $("fileInput").click());
dz.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " "){ e.preventDefault(); $("fileInput").click(); }});
$("pickBtn").addEventListener("click", e => { e.stopPropagation(); $("fileInput").click(); });
$("fileInput").addEventListener("change", e => e.target.files[0] && setFile(e.target.files[0]));
["dragenter","dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave","drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", e => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
$("removeBtn").addEventListener("click", removeFile);

$("sampleBtn").addEventListener("click", async () => {
  try { const b = await (await fetch("/api/sample")).blob(); setFile(new File([b], "sample_note.png", {type: "image/png"})); }
  catch { showError("Could not load the sample note."); }
});

$("recognizeBtn").addEventListener("click", async () => {
  if (!currentFile) return;
  showError(""); show($("loading"), true); $("recognizeBtn").disabled = true;
  const fd = new FormData(); fd.append("image", currentFile); fd.append("method", $("methodSel").value);
  const t0 = performance.now();
  const tick = () => { $("statusTitle").textContent = `Reading your handwriting… ${Math.round((performance.now()-t0)/1000)}s`; };
  tick(); const timer = setInterval(tick, 1000);
  try {
    const res = await fetch("/api/recognize", {method: "POST", body: fd});
    const data = await res.json().catch(() => ({error: "The server sent an unreadable response."}));
    if (!res.ok) throw new Error(data.error || "Recognition failed.");
    renderResult(data);
  } catch (e) { showError(e.message); }
  finally { clearInterval(timer); show($("loading"), false); $("recognizeBtn").disabled = false; }
});

function renderResult(d){
  $("resultImg").src = previewURL;
  $("textOut").value = d.text || "";
  const conf = d.avg_confidence != null ? ` · average confidence ${(d.avg_confidence*100).toFixed(0)}%` : "";
  $("statusBar").textContent = `${d.warning ? "Finished with a warning" : "Recognition complete"} · ${d.seconds}s · ${d.lines.length} lines · ${d.method === "word-detector" ? "word detector" : "line mode"}${conf}`;
  const low = d.lines.filter(l => l.confidence < 0.5), box = $("uncertain");
  box.textContent = "";
  if (d.warning){ box.textContent = d.warning; show(box, true); }
  else if (low.length){
    const s = document.createElement("strong"); s.textContent = "Check these lines (low confidence):";
    const ul = document.createElement("ul"); ul.className = "mb-0";
    low.forEach(l => { const li = document.createElement("li"); li.textContent = `${l.text} (${(l.confidence*100).toFixed(0)}%)`; ul.appendChild(li); });
    box.append(s, ul); show(box, true);
  } else show(box, false);
  show($("results"), true);
  $("results").scrollIntoView({behavior: "smooth"});
}

function flash(btn, msg){ const o = btn.innerHTML; btn.innerHTML = `<i class="bi bi-check2 me-1"></i>${msg}`; setTimeout(() => btn.innerHTML = o, 1400); }
$("copyBtn").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("textOut").value); }
  catch { $("textOut").select(); document.execCommand("copy"); }
  flash($("copyBtn"), "Copied");
});
$("clearBtn").addEventListener("click", () => { $("textOut").value = ""; $("textOut").focus(); });
$("txtBtn").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([$("textOut").value], {type: "text/plain;charset=utf-8"}));
  a.download = "inktotext-notes.txt"; a.click(); URL.revokeObjectURL(a.href);
});
$("pdfBtn").addEventListener("click", () => {
  if (!window.jspdf){ showError("The PDF library could not load. Check your internet connection."); return; }
  const doc = new window.jspdf.jsPDF({unit: "pt", format: "a4"});
  const m = 56, w = doc.internal.pageSize.getWidth() - m*2, ph = doc.internal.pageSize.getHeight();
  doc.setFont("helvetica", "bold"); doc.setFontSize(16); doc.text("InkToText AI - Recognized Notes", m, m);
  doc.setFont("helvetica", "normal"); doc.setFontSize(11);
  let y = m + 30;
  doc.splitTextToSize($("textOut").value || " ", w).forEach(line => {
    if (y > ph - m){ doc.addPage(); y = m; }
    doc.text(line, m, y); y += 16;
  });
  doc.save("inktotext-notes.pdf");
});
$("resetBtn").addEventListener("click", () => {
  removeFile(); show($("results"), false); $("textOut").value = "";
  $("upload").scrollIntoView({behavior: "smooth"});
});
