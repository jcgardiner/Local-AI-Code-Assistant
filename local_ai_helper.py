"""
local_ai_helper.py
------------------
A private AI helper GUI that talks to your LOCAL Ollama model.

Features:
  - Chat UI in your browser (everything stays on your machine).
  - Attachments (paperclip): images go to a vision model, text/code files
    are read and added to your message.
  - Side menu of saved chats: hover to see the full first message,
    click to reopen. Chats are saved as JSON files next to this script.
  - REDACTION hook that runs on your text before it's sent to the model.
  - API-FALLBACK toggle stub (off by default).

Settings live in config.toml next to this script (models, system prompt,
history length). Copy config.example.toml to config.toml to start.
Anything missing from it uses the defaults below.
Needs Python 3.11+ to read the config file.

Setup (one time):
    pip install -U gradio requests
    ollama pull qwen2.5-coder:3b     # text / code
    ollama pull qwen2.5vl:3b         # images (optional)

Run:
    python local_ai_helper.py
Then open the URL it prints (usually http://127.0.0.1:7860).
"""

import base64
import html
import inspect
import json
import os
import shutil
import time
import uuid
from pathlib import Path

# Turn off Gradio's usage statistics (on by default). This has to be set
# before gradio is imported, because gradio reads it while loading.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import requests

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

# Used for any setting config.toml doesn't provide.
DEFAULTS = {
    "ollama_url": "http://localhost:11434",
    "text_model": "qwen2.5-coder:3b",
    "vision_model": "qwen2.5vl:3b",  # "" turns off image support
    "system_prompt": (
        "You are a concise, helpful coding and documentation assistant. "
        "When asked for SQL, use the schema provided in the message if present."
    ),
    # On CPU, re-sending the whole transcript each turn gets slow fast.
    # 6 entries = roughly the last 3 exchanges.
    "max_history": 6,
    # Attached text files longer than this are cut off.
    "max_text_chars": 12000,
}


def load_config():
    """Start from DEFAULTS and apply whatever config.toml sets."""
    settings = dict(DEFAULTS)
    if not CONFIG_PATH.exists():
        print(f"No config file at {CONFIG_PATH}. Using built-in defaults.")
        print("To change settings, copy config.example.toml to config.toml and edit it.")
        return settings
    try:
        import tomllib  # built into Python 3.11 and newer
    except ModuleNotFoundError:
        print("Reading config.toml needs Python 3.11 or newer. Using built-in defaults.")
        return settings
    try:
        with open(CONFIG_PATH, "rb") as f:
            user_settings = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"config.toml couldn't be read, so it was ignored: {e}")
        return settings

    for key, value in user_settings.items():
        if key not in DEFAULTS:
            print(f"config.toml: unknown setting '{key}' was ignored.")
        elif type(value) is not type(DEFAULTS[key]):
            expected = "a whole number" if isinstance(DEFAULTS[key], int) else "text in quotes"
            print(f"config.toml: '{key}' should be {expected}. Using the default instead.")
        else:
            settings[key] = value
    return settings


CONFIG = load_config()
OLLAMA_BASE = CONFIG["ollama_url"].rstrip("/").removesuffix("/api/chat")
TEXT_MODEL = CONFIG["text_model"]
VISION_MODEL = CONFIG["vision_model"]
SYSTEM_PROMPT = CONFIG["system_prompt"].strip()
MAX_HISTORY = CONFIG["max_history"]
MAX_TEXT_CHARS = CONFIG["max_text_chars"]

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
TEXT_EXT = {
    ".txt", ".md", ".sql", ".py", ".csv", ".json", ".js", ".ts", ".html",
    ".css", ".yaml", ".yml", ".log", ".xml", ".ini", ".cfg", ".toml", ".sh",
    ".ps1", ".bat",
}

DATA_DIR = Path(__file__).resolve().parent / "chat_history"
ATTACH_DIR = DATA_DIR / "attachments"
DATA_DIR.mkdir(exist_ok=True)
ATTACH_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# 1) REDACTION HOOK
#    Runs on user text before it's sent to the model. If you already
#    added your own rules to the old file, paste them in here.
# ----------------------------------------------------------------------
def redact(text: str) -> str:
    return text


# ----------------------------------------------------------------------
# 2) BACKENDS
# ----------------------------------------------------------------------
def call_local(messages, model):
    """Stream a reply from Ollama, yielding text pieces."""
    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=600,
        ) as resp:
            if resp.status_code != 200:
                # Show Ollama's actual reason instead of a bare "400".
                yield f"[Ollama error {resp.status_code}: {resp.text}]"
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    yield f"[Ollama error: {data['error']}]"
                    return
                yield data.get("message", {}).get("content", "")
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield f"[Can't reach Ollama at {OLLAMA_BASE}. Start Ollama and try again.]"
    except Exception as e:
        yield f"[Error contacting Ollama: {e}]"


def installed_models():
    """Ask Ollama which models are installed.

    Returns (all_models, image_models). Both are empty if Ollama isn't running.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        names = sorted(m["name"] for m in resp.json().get("models", []))
    except Exception:
        return [], []

    image_models = []
    reported = False  # older Ollama versions don't report capabilities
    for name in names:
        try:
            info = requests.post(f"{OLLAMA_BASE}/api/show", json={"model": name}, timeout=5).json()
        except Exception:
            continue
        if "capabilities" in info:
            reported = True
            if "vision" in info["capabilities"]:
                image_models.append(name)
    return names, (image_models if reported else names)


def call_api(messages):
    """Stub for a Zero-Data-Retention API. Paste your version in here."""
    yield "[API fallback isn't set up yet. Turn off the checkbox to use the local model.]"


# ----------------------------------------------------------------------
# 3) SAVED CHATS (one JSON file per chat in ./chat_history)
#    Each message: {"role", "content" (plain string), "files": [paths]}
#    We keep our own copy of the conversation instead of reading Gradio's
#    history back, so Gradio's format changes can't break Ollama requests.
# ----------------------------------------------------------------------
def new_chat_obj():
    return {"id": uuid.uuid4().hex[:12], "title": "", "updated": time.time(), "messages": []}


def chat_path(chat_id):
    return DATA_DIR / f"{chat_id}.json"


def save_chat(chat):
    chat["updated"] = time.time()
    chat_path(chat["id"]).write_text(json.dumps(chat, indent=2), encoding="utf-8")


def load_chat(chat_id):
    try:
        return json.loads(chat_path(chat_id).read_text(encoding="utf-8"))
    except Exception:
        return None


def list_chats():
    chats = []
    for f in DATA_DIR.glob("*.json"):
        try:
            chats.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    chats.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return chats


def sidebar_update(selected_id=None):
    choices = [
        ((c.get("title") or "Untitled").replace("\n", " "), c["id"])
        for c in list_chats()
    ]
    ids = {cid for _, cid in choices}
    return gr.Radio(choices=choices, value=selected_id if selected_id in ids else None)


def save_uploads(files, chat_id):
    """Copy uploaded files out of Gradio's temp cache so saved chats keep them."""
    saved = []
    folder = ATTACH_DIR / chat_id
    folder.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        if isinstance(f, str):
            src = f
        elif isinstance(f, dict):
            src = f.get("path")
        else:
            src = getattr(f, "path", None)
        if not src or not Path(src).exists():
            continue
        dest = folder / f"{uuid.uuid4().hex[:6]}_{Path(src).name}"
        shutil.copy(src, dest)
        saved.append(str(dest))
    return saved


# ----------------------------------------------------------------------
# 4) BUILDING WHAT GETS SENT / SHOWN
# ----------------------------------------------------------------------
def build_ollama_messages(messages, allow_images):
    """Turn saved messages into Ollama's format. Returns (messages, has_images)."""
    out = [{"role": "system", "content": SYSTEM_PROMPT}]
    has_images = False
    for m in messages[-MAX_HISTORY:]:
        content = m.get("content", "")
        images = []
        for p in m.get("files", []):
            path = Path(p)
            name = path.name.split("_", 1)[-1]
            if not path.exists():
                content += f"\n\n[Attachment {name} is missing from disk.]"
            elif path.suffix.lower() in IMAGE_EXT:
                images.append(base64.b64encode(path.read_bytes()).decode())
            elif path.suffix.lower() in TEXT_EXT:
                body = path.read_text(encoding="utf-8", errors="replace")
                if len(body) > MAX_TEXT_CHARS:
                    body = body[:MAX_TEXT_CHARS] + "\n[...file truncated...]"
                content += f"\n\n--- Attached file: {name} ---\n{body}"
            else:
                content += f"\n\n[Attached file {name} wasn't read: unsupported file type.]"
        if m["role"] == "user":
            content = redact(content)
        msg = {"role": m["role"], "content": content}
        if images:
            if allow_images:
                msg["images"] = images
                has_images = True
            else:
                msg["content"] += "\n\n[An image was attached but no image model is selected.]"
        out.append(msg)
    return out, has_images


def to_display(messages):
    """Turn saved messages into what the Chatbot shows."""
    out = []
    for i, m in enumerate(messages):
        for p in m.get("files", []):
            if Path(p).exists():
                out.append({"role": m["role"], "content": {"path": p}})
        if m.get("content"):
            out.append({"role": m["role"], "content": m["content"]})
        # A question with no reply after it would be drawn in the same bubble
        # as the next question, which throws off the question panel.
        if m["role"] == "user" and i + 1 < len(messages):
            nxt = messages[i + 1]
            if nxt["role"] != "assistant" or not nxt.get("content"):
                out.append({"role": "assistant", "content": "*No reply was saved for this question.*"})
    return out


# ----------------------------------------------------------------------
# 5) QUESTION PANEL (right side)
#    Lists every question in the open chat. Clicking one scrolls the chat
#    to it. Python builds the list as HTML; the scrolling happens in the
#    browser, using the JavaScript attached to each button (JUMP_JS).
# ----------------------------------------------------------------------
# Runs in the browser when a question is clicked. "this" is the clicked
# button; its data-q attribute holds the question's position in the chat.
JUMP_JS = """
const rows = document.querySelectorAll('#chatbot .user-row');
const row = rows[Number(this.dataset.q)];
if (!row) {
    console.warn('Question panel: no message found for question', this.dataset.q, '- found', rows.length);
    return;
}
// Find the chat's own scrolling box, so only the chat moves, not the page.
let box = row.parentElement;
while (box && box.id !== 'chatbot' && box.scrollHeight <= box.clientHeight) {
    box = box.parentElement;
}
if (!box || box.id === 'chatbot') box = row.closest('#chatbot');
const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const gap = 44;  // room for the chat's toolbar icons
const top = row.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - gap;
box.scrollTo({ top: top, behavior: calm ? 'auto' : 'smooth' });
row.classList.remove('q-flash');
void row.offsetWidth;
row.classList.add('q-flash');
setTimeout(() => row.classList.remove('q-flash'), 1200);
"""
JUMP_ATTR = html.escape(JUMP_JS, quote=True)


def questions_html(chat):
    items = []
    n = 0
    for m in chat["messages"]:
        if m["role"] != "user":
            continue
        label = m.get("content") or ", ".join(
            Path(p).name.split("_", 1)[-1] for p in m.get("files", [])
        ) or "(empty message)"
        items.append(
            f'<button type="button" class="q-item" data-q="{n}" onclick="{JUMP_ATTR}">'
            f'<span class="q-num">{n + 1}</span>'
            f'<span class="q-text">{html.escape(label)}</span>'
            f"</button>"
        )
        n += 1
    if not items:
        return '<p class="q-empty">Questions you ask in this chat will show up here.</p>'
    return '<nav class="q-list">' + "".join(items) + "</nav>"


# ----------------------------------------------------------------------
# 6) EVENT HANDLERS
#    Every handler returns one value per output, in the same order as
#    the outputs list where it's wired up at the bottom of the file.
# ----------------------------------------------------------------------
def respond(msg, chat, use_api, text_model, image_model):
    msg = msg or {}
    text = (msg.get("text") or "").strip()
    files = save_uploads(msg.get("files"), chat["id"])
    if not text and not files:
        yield gr.update(), chat, gr.update(), gr.update(), gr.update()
        return

    chat["messages"].append({"role": "user", "content": text, "files": files})
    if not chat["title"]:
        chat["title"] = text or Path(files[0]).name.split("_", 1)[-1]
    save_chat(chat)

    display = to_display(chat["messages"])
    display.append({"role": "assistant", "content": ""})
    yield (
        display,
        chat,
        gr.MultimodalTextbox(value=None, interactive=False),
        sidebar_update(chat["id"]),
        questions_html(chat),
    )

    payload, has_images = build_ollama_messages(chat["messages"], allow_images=bool(image_model))
    if use_api:
        stream = call_api(payload)
    else:
        stream = call_local(payload, image_model if has_images else text_model)

    reply = ""
    try:
        for piece in stream:
            reply += piece
            display[-1]["content"] = reply
            yield display, chat, gr.update(), gr.update(), gr.update()
    finally:
        # Runs even if the answer is cut off (page refreshed or closed
        # mid-answer), so whatever arrived still gets saved.
        chat["messages"].append({"role": "assistant", "content": reply, "files": []})
        save_chat(chat)
    yield (
        display,
        chat,
        gr.MultimodalTextbox(value=None, interactive=True),
        sidebar_update(chat["id"]),
        questions_html(chat),
    )


def open_chat(chat_id):
    chat = load_chat(chat_id) if chat_id else None
    if chat is None:
        chat = new_chat_obj()
    return to_display(chat["messages"]), chat, questions_html(chat)


def start_new_chat():
    chat = new_chat_obj()
    return [], chat, sidebar_update(None), questions_html(chat)


def delete_chat(chat):
    chat_path(chat["id"]).unlink(missing_ok=True)
    shutil.rmtree(ATTACH_DIR / chat["id"], ignore_errors=True)
    return start_new_chat()


def on_load():
    # Reopen the most recent saved chat; start a new one only if none exist.
    chats = list_chats()
    chat = chats[0] if chats else new_chat_obj()
    return (
        to_display(chat["messages"]),
        chat,
        sidebar_update(chat["id"]),
        questions_html(chat),
    )


NO_IMAGE_MODEL = ("None (skip images)", "")


def model_dropdowns():
    """Fill both dropdowns with installed models, preselecting config.toml's choices."""
    names, image_names = installed_models()
    text_choices = list(names)
    if TEXT_MODEL not in text_choices:
        text_choices.insert(0, TEXT_MODEL)
    image_choices = list(image_names)
    if VISION_MODEL and VISION_MODEL not in image_choices:
        image_choices.insert(0, VISION_MODEL)
    return (
        gr.Dropdown(choices=text_choices, value=TEXT_MODEL),
        gr.Dropdown(choices=[NO_IMAGE_MODEL] + [(n, n) for n in image_choices], value=VISION_MODEL),
    )


# ----------------------------------------------------------------------
# 7) LAYOUT
# ----------------------------------------------------------------------
CSS = """
/* Left: saved chats */
#chat-list { max-height: 72vh; overflow-y: auto; }
#chat-list .wrap { display: flex; flex-direction: column; gap: 2px; }
#chat-list input[type="radio"] { display: none; }
#chat-list label {
    display: block; width: 100%; box-sizing: border-box;
    padding: 8px 10px; border-radius: 8px; cursor: pointer;
    background: transparent; border: none; box-shadow: none;
}
#chat-list label span {
    display: block; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; margin: 0;
}
#chat-list label:hover { background: var(--background-fill-secondary); }
#chat-list label:hover span { white-space: normal; overflow: visible; }
#chat-list label.selected { background: var(--color-accent-soft); }

/* Right: questions in this chat */
#question-panel .q-list {
    display: flex; flex-direction: column; gap: 2px;
    max-height: 72vh; overflow-y: auto;
}
#question-panel .q-item {
    display: flex; gap: 8px; align-items: baseline;
    width: 100%; padding: 8px 10px; border: none; border-radius: 8px;
    background: transparent; color: var(--body-text-color);
    font: inherit; text-align: left; cursor: pointer;
}
#question-panel .q-item:hover { background: var(--background-fill-secondary); }
#question-panel .q-item:focus-visible { outline: 2px solid var(--color-accent); }
#question-panel .q-num { flex: none; color: var(--body-text-color-subdued); }
#question-panel .q-text {
    min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#question-panel .q-item:hover .q-text,
#question-panel .q-item:focus-visible .q-text { white-space: normal; }
#question-panel .q-empty { color: var(--body-text-color-subdued); padding: 8px 10px; }

/* One-screen layout: the page never scrolls; only the chat does. */
html, body { height: 100%; overflow: hidden; }
#mid-col {
    height: calc(100vh - 76px);   /* window height minus the page's top/bottom spacing */
    display: flex; flex-direction: column; flex-wrap: nowrap;
}
#mid-col > * { flex: none !important; }    /* header and message box: natural height */
#mid-col > #chatbot {
    flex: 1 1 0 !important; min-height: 0;   /* chat: take whatever height is left over */
    height: auto !important;
}

/* Brief outline on the question you jumped to */
#chatbot { isolation: isolate; }
#chatbot .user-row.q-flash .message { animation: q-flash 1.2s ease-out; }
@keyframes q-flash {
    from { box-shadow: 0 0 0 3px var(--color-accent); }
    to   { box-shadow: 0 0 0 3px transparent; }
}
@media (prefers-reduced-motion: reduce) {
    #chatbot .user-row.q-flash .message { animation: none; outline: 2px solid var(--color-accent); }
}
"""

# Gradio 6 takes css in launch(); Gradio 5 takes it in Blocks().
SETTINGS_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters
page_settings = {"css": CSS}
blocks_kwargs = {} if SETTINGS_IN_LAUNCH else page_settings

# Gradio 5 needs type="messages"; Gradio 6 removed the argument.
chatbot_kwargs = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {}

with gr.Blocks(**blocks_kwargs) as demo:
    chat_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1, min_width=220, elem_id="left-col"):
            new_btn = gr.Button("New chat", variant="primary")
            chat_list = gr.Radio(choices=[], show_label=False, elem_id="chat-list", container=False)
            delete_btn = gr.Button("Delete this chat", size="sm")

        with gr.Column(scale=4, elem_id="mid-col"):
            with gr.Row():
                text_model_dd = gr.Dropdown(
                    choices=[TEXT_MODEL], value=TEXT_MODEL, label="Model", interactive=True
                )
                image_model_dd = gr.Dropdown(
                    choices=[NO_IMAGE_MODEL] + ([(VISION_MODEL, VISION_MODEL)] if VISION_MODEL else []),
                    value=VISION_MODEL,
                    label="Model for images",
                    interactive=True,
                )
            use_api = gr.Checkbox(label="Use API fallback (off = fully local)", value=False)
            chatbot = gr.Chatbot(elem_id="chatbot", height=620, show_label=False, **chatbot_kwargs)
            msg_box = gr.MultimodalTextbox(
                file_count="multiple",
                placeholder="Ask something, or attach an image or file with the paperclip",
                show_label=False,
            )

        with gr.Column(scale=1, min_width=220, elem_id="right-col"):
            gr.Markdown("**In this chat**")
            question_panel = gr.HTML(elem_id="question-panel")

    msg_box.submit(
        respond,
        [msg_box, chat_state, use_api, text_model_dd, image_model_dd],
        [chatbot, chat_state, msg_box, chat_list, question_panel],
    )
    chat_list.input(open_chat, chat_list, [chatbot, chat_state, question_panel])
    new_btn.click(start_new_chat, None, [chatbot, chat_state, chat_list, question_panel])
    delete_btn.click(delete_chat, chat_state, [chatbot, chat_state, chat_list, question_panel])
    demo.load(on_load, None, [chatbot, chat_state, chat_list, question_panel])
    demo.load(model_dropdowns, None, [text_model_dd, image_model_dd])

if __name__ == "__main__":
    launch_kwargs = {"allowed_paths": [str(DATA_DIR)]}  # lets the chat show saved attachments
    if SETTINGS_IN_LAUNCH:
        launch_kwargs.update(page_settings)
    demo.launch(**launch_kwargs)
