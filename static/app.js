/* PDFChat frontend — vanilla JS, no framework. */
(() => {
  const $ = (id) => document.getElementById(id);
  const apiStatus = $("apiStatus"), statusText = $("statusText");
  const settingsBtn = $("settingsBtn"), settingsModal = $("settingsModal");
  const settingsClose = $("settingsClose"), keyInput = $("keyInput");
  const keySave = $("keySave"), keySpinner = $("keySpinner"), keyRemove = $("keyRemove");
  const keyState = $("keyState"), keyError = $("keyError"), keyOk = $("keyOk");
  const docEmpty = $("docEmpty"), docCard = $("docCard");
  const docName = $("docName"), docSub = $("docSub"), docPages = $("docPages");
  const docSize = $("docSize"), docChunks = $("docChunks"), docStatus = $("docStatus");
  const removeDocBtn = $("removeDocBtn");
  const uploadHero = $("uploadHero"), dropzone = $("dropzone"), fileInput = $("fileInput");
  const browseBtn = $("browseBtn"), uploadError = $("uploadError");
  const processing = $("processing"), procBarFill = $("procBarFill");
  const chatPanel = $("chatPanel"), chatDocName = $("chatDocName"), modelTag = $("modelTag");
  const messages = $("messages"), composer = $("composer"), question = $("question");
  const sendBtn = $("sendBtn"), chatError = $("chatError");

  let MAX_MB = 25;
  let keySource = null;
  let statusTimer = null;
  let hasDoc = false;

  function show(el) { el.hidden = false; }
  function hide(el) { el.hidden = true; }
  function setUploadError(msg) {
    if (!msg) { hide(uploadError); uploadError.textContent = ""; return; }
    uploadError.textContent = msg; show(uploadError);
  }
  function setChatError(msg) {
    if (!msg) { hide(chatError); chatError.textContent = ""; return; }
    chatError.textContent = msg; show(chatError);
  }
  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  /* ---------- status + document restore ---------- */
  function pillText(configured) {
    const narrow = window.innerWidth < 560;
    if (configured) return "API connected";
    return narrow ? "API key missing" : "API key missing — open Settings";
  }
  async function loadStatus() {
    apiStatus.classList.add("checking");
    apiStatus.classList.remove("ok", "bad");
    try {
      const r = await fetch("/api/status");
      const d = await r.json();
      apiStatus.classList.remove("checking");
      keySource = d.source || null;
      if (typeof d.max_pdf_mb === "number" && d.max_pdf_mb > 0) {
        MAX_MB = d.max_pdf_mb;
        const fl = $("formatsLine");
        if (fl) fl.textContent = "PDF only · up to " + (Number.isInteger(MAX_MB) ? MAX_MB : MAX_MB.toFixed(1)) + "MB · text-based (no scans)";
        const ms = $("maxMbSide");
        if (ms) ms.textContent = (Number.isInteger(MAX_MB) ? MAX_MB : MAX_MB.toFixed(1)) + "MB";
      }
      if (d.configured) {
        apiStatus.classList.add("ok");
        statusText.textContent = pillText(true);
        statusText.title = d.message || "";
      } else {
        apiStatus.classList.add("bad");
        statusText.textContent = pillText(false);
        statusText.title = d.message || "Add your Groq API key via Settings or .env";
      }
      renderKeyState(d);
      if (d.document && d.document.loaded) renderDoc(d.document);
    } catch {
      apiStatus.classList.remove("checking");
      apiStatus.classList.add("bad");
      statusText.textContent = "Server unreachable";
    }
  }
  window.addEventListener("resize", () => {
    clearTimeout(statusTimer);
    statusTimer = setTimeout(loadStatus, 250);
  });

  /* ---------- settings modal ---------- */
  function renderKeyState(d) {
    if (!d) return;
    keyState.classList.remove("on", "off");
    if (d.configured) {
      keyState.classList.add("on");
      const where = d.source === "settings"
        ? "Key active — entered via Settings (this session only)."
        : "Key active — loaded from your .env file.";
      keyState.textContent = "✓ " + where + " Model: " + (d.model || "ready") + ".";
    } else {
      keyState.classList.add("off");
      keyState.textContent = "✕ No API key configured yet.";
    }
    keyRemove.disabled = keySource !== "settings";
    keyRemove.title = keySource === "settings"
      ? "Forget the key entered via Settings"
      : "There is no Settings key to remove (the .env key, if any, stays)";
  }
  function setKeyError(msg) {
    if (!msg) { hide(keyError); keyError.textContent = ""; return; }
    keyError.textContent = msg; show(keyError);
  }
  function setKeyOk(msg) {
    if (!msg) { hide(keyOk); keyOk.textContent = ""; return; }
    keyOk.textContent = msg; show(keyOk);
  }
  function openSettings() {
    setKeyError(""); setKeyOk(""); keyInput.value = "";
    show(settingsModal);
    loadStatus();
    setTimeout(() => keyInput.focus(), 50);
  }
  function closeSettings() { hide(settingsModal); settingsBtn.focus(); }
  settingsBtn.addEventListener("click", openSettings);
  settingsClose.addEventListener("click", closeSettings);
  settingsModal.addEventListener("click", (e) => { if (e.target === settingsModal) closeSettings(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !settingsModal.hidden) closeSettings();
  });
  keyInput.addEventListener("input", () => { setKeyError(""); setKeyOk(""); });
  $("keyForm").addEventListener("submit", (e) => { e.preventDefault(); keySave.click(); });
  keySave.addEventListener("click", async () => {
    const key = keyInput.value.trim();
    setKeyError(""); setKeyOk("");
    if (!key) { setKeyError("Please paste your Groq API key first."); keyInput.focus(); return; }
    keySave.disabled = true; show(keySpinner);
    try {
      const res = await fetch("/api/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setKeyError(data.error || ("Could not save the key (HTTP " + res.status + ").")); return; }
      keyInput.value = "";
      setKeyOk("✓ " + (data.message || "Key saved."));
      await loadStatus();
    } catch {
      setKeyError("Could not reach the server. Make sure the app is running and try again.");
    } finally {
      keySave.disabled = false; hide(keySpinner);
    }
  });
  keyRemove.addEventListener("click", async () => {
    setKeyError(""); setKeyOk("");
    try {
      const res = await fetch("/api/key", { method: "DELETE" });
      const data = await res.json().catch(() => ({}));
      setKeyOk("✓ " + (data.message || "Key removed."));
      await loadStatus();
    } catch {
      setKeyError("Could not reach the server. Try again.");
    }
  });

  /* ---------- document rendering ---------- */
  function renderDoc(meta) {
    hasDoc = true;
    hide(docEmpty); show(docCard);
    docName.textContent = meta.filename || "document.pdf";
    docName.title = meta.filename || "";
    docSub.textContent = meta.size || "";
    docPages.textContent = meta.pages != null ? meta.pages : "—";
    docSize.textContent = meta.size || "—";
    docChunks.textContent = meta.chunks != null ? meta.chunks : "—";
    docStatus.textContent = "Ready";
    chatDocName.textContent = meta.filename || "your document";
    hide(uploadHero); show(chatPanel);
  }
  function clearDocUI() {
    hasDoc = false;
    show(docEmpty); hide(docCard);
    show(uploadHero); hide(chatPanel);
    messages.innerHTML = "";
    fileInput.value = "";
  }

  /* ---------- upload with staged progress ---------- */
  function setStep(name) {
    document.querySelectorAll(".proc-step").forEach((el) => {
      el.classList.remove("active", "done");
      if (el.dataset.step === name) el.classList.add("active");
    });
  }
  function markDoneUpto(name) {
    const order = ["upload", "extract", "index", "ready"];
    const idx = order.indexOf(name);
    document.querySelectorAll(".proc-step").forEach((el) => {
      const i = order.indexOf(el.dataset.step);
      el.classList.toggle("done", i < idx);
      el.classList.toggle("active", el.dataset.step === name);
    });
  }
  async function uploadFile(file) {
    setUploadError("");
    if (!file) return;
    const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
    if (!isPdf) {
      setUploadError("Unsupported file type '" + file.name + "'. Please upload a PDF file (.pdf).");
      return;
    }
    const mb = file.size / (1024 * 1024);
    if (mb > MAX_MB) {
      setUploadError("PDF is too large (" + mb.toFixed(1) + "MB — limit is " + MAX_MB + "MB). Please split it or choose a smaller file.");
      return;
    }
    show(processing);
    setStep("upload");
    procBarFill.style.width = "12%";
    const stageTimer = setInterval(() => {
      // Advance the displayed stage while the server works (real phases,
      // single request). Timers only move the indicator, never fake results.
      const w = parseFloat(procBarFill.style.width) || 12;
      if (w < 30) { markDoneUpto("upload"); procBarFill.style.width = "28%"; }
      else if (w < 60) { markDoneUpto("extract"); procBarFill.style.width = "58%"; }
      else if (w < 88) { markDoneUpto("index"); procBarFill.style.width = "86%"; }
    }, 900);

    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      clearInterval(stageTimer);
      if (!res.ok) {
        hide(processing);
        setUploadError(data.error || ("Upload failed (HTTP " + res.status + "). Please try again."));
        return;
      }
      markDoneUpto("ready");
      procBarFill.style.width = "100%";
      await new Promise((r) => setTimeout(r, 450));
      hide(processing);
      procBarFill.style.width = "8%";
      renderDoc(data);
      addMessage("ai", "I've read **" + data.filename + "** (" + data.pages + " pages, " + data.chunks + " passages indexed). Ask me anything about it — I'll answer only from the document and show page sources.", []);
      if (data.model) { /* model shown per answer instead */ }
    } catch {
      clearInterval(stageTimer);
      hide(processing);
      setUploadError("Could not reach the server. Make sure the app is running and try again.");
    }
  }

  browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });
  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });

  removeDocBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/document", { method: "DELETE" });
    } catch { /* local UI resets regardless */ }
    clearDocUI();
    setUploadError("");
  });

  /* ---------- chat ---------- */
  function renderLite(text) {
    // Fenced code blocks first (```lang ... ```), then inline markdown.
    const parts = escapeHtml(text).split(/```/);
    let html = "";
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 1) {
        const code = parts[i].replace(/^\s*[a-zA-Z0-9+#-]+\n/, "").replace(/^\n/, "");
        html += "<pre><code>" + code + "</code></pre>";
      } else {
        html += renderBody(parts[i]);
      }
    }
    return html;
  }
  function renderBody(escaped) {
    const lines = escaped.split("\n");
    let html = "", inList = false;
    for (const line of lines) {
      const t = line.trim();
      if (/^([-*•]\s+)/.test(t)) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += "<li>" + t.replace(/^([-*•]\s+)/, "") + "</li>";
      } else if (/^\d+\.\s+/.test(t)) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += "<li>" + t.replace(/^\d+\.\s+/, "") + "</li>";
      } else {
        if (inList) { html += "</ul>"; inList = false; }
        if (t === "") html += "<br>";
        else html += "<p style='margin:.4em 0'>" + t + "</p>";
      }
    }
    if (inList) html += "</ul>";
    return html
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[p\.\s*(\d+)\]/g, "[p. $1]");
  }
  function scrollChat() { messages.scrollTop = messages.scrollHeight; }
  function addMessage(kind, text, sources) {
    const div = document.createElement("div");
    if (kind === "user") {
      div.className = "msg user";
      div.textContent = text;
    } else if (kind === "error") {
      div.className = "msg msg-error";
      div.textContent = text;
    } else {
      div.className = "msg ai";
      div.innerHTML = renderLite(text);
      if (sources && sources.length) {
        const s = document.createElement("div");
        s.className = "sources";
        const label = document.createElement("span");
        label.className = "src-label";
        label.textContent = "Sources";
        s.appendChild(label);
        sources.forEach((p) => {
          const c = document.createElement("span");
          c.className = "src-chip";
          c.textContent = "p. " + p;
          s.appendChild(c);
        });
        div.appendChild(s);
      }
    }
    messages.appendChild(div);
    scrollChat();
    return div;
  }

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      question.value = chip.dataset.q;
      question.focus();
    });
  });

  composer.addEventListener("submit", async (e) => {
    e.preventDefault();
    setChatError("");
    const q = question.value.trim();
    if (!q) { setChatError("Please type a question first."); question.focus(); return; }
    if (!hasDoc) { setChatError("No document loaded. Upload a PDF first, then ask your question."); return; }
    question.value = "";
    addMessage("user", q);
    sendBtn.disabled = true;
    const typing = document.createElement("div");
    typing.className = "msg ai typing";
    typing.innerHTML = "<i></i><i></i><i></i>";
    typing.setAttribute("aria-label", "PDFChat is thinking");
    messages.appendChild(typing);
    scrollChat();
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json().catch(() => ({}));
      typing.remove();
      if (!res.ok) {
        addMessage("error", data.error || ("Request failed (HTTP " + res.status + "). Please try again."));
        return;
      }
      addMessage("ai", data.answer || "(empty response)", data.sources || []);
      if (data.model) {
        modelTag.textContent = "◈ " + data.model;
        modelTag.hidden = false;
      }
    } catch {
      typing.remove();
      addMessage("error", "Could not reach the server. Make sure the app is running and try again.");
    } finally {
      sendBtn.disabled = false;
      question.focus();
    }
  });

  /* ---------- init ---------- */
  loadStatus();
})();
