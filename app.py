import os
os.makedirs(r"D:\hf\xet", exist_ok=True)
os.makedirs(r"D:\tmp", exist_ok=True)
os.makedirs(r"D:\whisper-models", exist_ok=True)
os.environ["HF_HOME"] = r"D:\hf"
os.environ["HF_XET_CACHE"] = r"D:\hf\xet"
os.environ["HF_HUB_CACHE"] = r"D:\hf\hub"
os.environ["TMP"] = r"D:\tmp"
os.environ["TEMP"] = r"D:\tmp"
import json, tempfile, os, datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import ollama
from faster_whisper import WhisperModel
from pydantic import BaseModel

class OrderLine(BaseModel):
    customer: str = ""
    item: str = ""
    quantity: float = 1
    unit: str = ""
    delivery_date: str = ""
    address: str = ""
    notes: str = ""

class OrderList(BaseModel):
    orders: list[OrderLine]

st.set_page_config(page_title="Voice Order Taker", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
h1.title{font-size:2.4rem;margin:0;background:linear-gradient(90deg,#2DD4BF,#818CF8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800}
p.sub{color:#93A4C0;margin-top:.3rem;font-size:1.05rem}
.card{background:#121C30;border:1px solid #1E2B45;border-radius:16px;padding:1.1rem 1.3rem;margin-bottom:1rem}
.stat{font-size:1.9rem;font-weight:700;color:#2DD4BF}
.lbl{color:#93A4C0;font-size:.85rem;text-transform:uppercase;letter-spacing:.08em}
div.stButton>button{border-radius:12px;font-weight:600;border:0;
  background:linear-gradient(90deg,#14B8A6,#6366F1);color:white;padding:.6rem 1.4rem}
</style>
""", unsafe_allow_html=True)

def layers(css, n=7, depth=36):
    out = ""
    for i in range(n):
        z = -depth/2 + i*depth/(n-1)
        out += f'<div class="{css}" style="transform:translateZ({z}px)"></div>'
    return out

ROBOT = f"""
<style>
body{{margin:0;background:transparent;overflow:hidden}}
.scene{{perspective:900px;width:240px;height:250px;margin:0 auto;position:relative}}
.robot{{position:absolute;inset:0;transform-style:preserve-3d;
  animation:sway 7s ease-in-out infinite, bob 3s ease-in-out infinite}}
.part{{position:absolute;transform-style:preserve-3d}}
.head{{left:55px;top:50px;width:130px;height:96px}}
.hl{{position:absolute;inset:0;border-radius:30px;
  background:linear-gradient(145deg,#F4F8FF,#B7C7E4)}}
.hl:nth-child(1),.hl:nth-child(2){{background:#7F92B8}}
.face{{position:absolute;left:14px;top:20px;width:102px;height:54px;border-radius:22px;
  background:linear-gradient(160deg,#0B1220,#1B2A4A);transform:translateZ(19px);
  box-shadow:inset 0 0 12px #000}}
.eye{{position:absolute;top:16px;width:20px;height:22px;border-radius:50%;
  background:#2DD4BF;box-shadow:0 0 14px #2DD4BF;animation:blink 4.5s infinite}}
.eye.l{{left:22px}} .eye.r{{right:22px}}
.smile{{position:absolute;left:41px;top:38px;width:20px;height:9px;
  border-bottom:3px solid #2DD4BF;border-radius:0 0 20px 20px}}
.ear{{position:absolute;top:32px;width:16px;height:34px;border-radius:8px;background:#6366F1;
  box-shadow:0 0 10px #6366F188}}
.ear.l{{left:-12px;transform:translateZ(0)}} .ear.r{{right:-12px;transform:translateZ(0)}}
.ant{{position:absolute;left:62px;top:-26px;width:6px;height:28px;background:#B7C7E4;transform:translateZ(0)}}
.ball{{position:absolute;left:-5px;top:-12px;width:16px;height:16px;border-radius:50%;
  background:#818CF8;box-shadow:0 0 16px #818CF8;animation:pulse 1.6s infinite}}
.body{{left:75px;top:152px;width:90px;height:62px}}
.bl{{position:absolute;inset:0;border-radius:22px;background:linear-gradient(145deg,#E3ECFA,#9FB2D6)}}
.bl:nth-child(1),.bl:nth-child(2){{background:#6F82A8}}
.chest{{position:absolute;left:30px;top:20px;width:30px;height:20px;border-radius:8px;
  background:#2DD4BF;box-shadow:0 0 14px #2DD4BF;transform:translateZ(19px);animation:pulse 2.2s infinite}}
.arm{{width:22px;height:22px;border-radius:50%;background:#6366F1;top:168px;box-shadow:0 0 10px #6366F155}}
.arm.l{{left:48px;animation:wave 2.4s ease-in-out infinite}} .arm.r{{left:170px}}
.shadow{{position:absolute;left:60px;top:232px;width:120px;height:14px;border-radius:50%;
  background:rgba(0,0,0,.45);filter:blur(6px);animation:sh 3s ease-in-out infinite}}
@keyframes sway{{0%,100%{{transform:rotateY(-22deg) rotateX(4deg)}}50%{{transform:rotateY(22deg) rotateX(-2deg)}}}}
@keyframes bob{{0%,100%{{margin-top:0}}50%{{margin-top:-10px}}}}
@keyframes blink{{0%,92%,100%{{transform:scaleY(1)}}95%{{transform:scaleY(.1)}}}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.45}}}}
@keyframes wave{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-14px)}}}}
@keyframes sh{{0%,100%{{transform:scale(1);opacity:.9}}50%{{transform:scale(.8);opacity:.6}}}}
</style>
<div class="scene">
 <div class="shadow"></div>
 <div class="robot">
  <div class="part head">
    {layers("hl")}
    <div class="face"><div class="eye l"></div><div class="eye r"></div><div class="smile"></div></div>
    <div class="ear l"></div><div class="ear r"></div>
    <div class="ant"><div class="ball"></div></div>
  </div>
  <div class="part body">{layers("bl", 5, 28)}<div class="chest"></div></div>
  <div class="part arm l"></div><div class="part arm r"></div>
 </div>
</div>
"""

@st.cache_resource(show_spinner="Loading speech model...")
def load_whisper(size):
    return WhisperModel(size, device="cpu", compute_type="int8",
                    download_root=r"D:\whisper-models")

def transcribe(upload, size, lang):
    suffix = os.path.splitext(upload.name)[1] or ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upload.getvalue())
        path = f.name
    try:
        segs, _ = load_whisper(size).transcribe(
            path, language=None if lang == "auto" else lang,
            vad_filter=True, beam_size=5)
        return " ".join(s.text.strip() for s in segs)
    finally:
        os.remove(path)
initial_prompt="chocolate cake, samosa, biryani, kilo, dozen, kal shaam, address",
 
def extract(text, model):
    today = datetime.date.today().strftime("%A, %d %B %Y")
    system = (
        "You extract orders from a transcribed voice note sent to a home business. "
        "The text may be Urdu script, Roman Urdu, Pashto or English, or a mix. "
        f"Today is {today}. Convert kal (tomorrow), parson (day after tomorrow) and aaj (today) "
        "into YYYY-MM-DD dates. "
        "Rules: create one entry per item ordered. Each quantity must belong to the item it is "
        "spoken with. Write item names in English (cake, samosa, biryani). "
        "Copy the customer name and address exactly as spoken. "
        "If something is not mentioned, leave it empty. Never invent items.\n\n"
        "Example input: Assalam o alaikum, mera naam Ali hai. Do kilo chocolate cake aur "
        "teen dozen samosay kal shaam tak chahiye, house 12 Jinnah Road.\n"
        "Example output: {\"orders\":["
        "{\"customer\":\"Ali\",\"item\":\"chocolate cake\",\"quantity\":2,\"unit\":\"kg\","
        "\"delivery_date\":\"TOMORROW\",\"address\":\"house 12 Jinnah Road\",\"notes\":\"evening\"},"
        "{\"customer\":\"Ali\",\"item\":\"samosa\",\"quantity\":3,\"unit\":\"dozen\","
        "\"delivery_date\":\"TOMORROW\",\"address\":\"house 12 Jinnah Road\",\"notes\":\"evening\"}]}"
    )
    r = ollama.chat(
        model=model,
        format=OrderList.model_json_schema(),
        options={"temperature": 0, "num_ctx": 2048},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    )
    try:
        data = OrderList.model_validate_json(r["message"]["content"])
        return [o.model_dump() for o in data.orders]
    except Exception:
        return []


# ---------- Header ----------
c1, c2 = st.columns([1, 3])
with c1:
    components.html(ROBOT, height=265)
with c2:
    st.markdown('<h1 class="title">Voice Order Taker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Drop in WhatsApp voice notes. Get a clean order sheet and a daily prep list. '
                'Everything runs on this computer, so customer details never leave it.</p>',
                unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    wsize = st.selectbox("Speech model", ["small", "medium", "base"], index=0,
                         help="Larger is more accurate for Urdu but slower.")
    lang = st.selectbox("Language", ["ur", "auto", "ps", "en"], index=0)
    llm = st.text_input("Ollama model", "gemma4:latest")
    st.caption("Offline. No API key. No per-order cost.")

if "orders" not in st.session_state:
    st.session_state.orders = pd.DataFrame()
    st.session_state.transcripts = {}

files = st.file_uploader("Voice notes", type=["ogg", "opus", "mp3", "m4a", "wav", "mpeg"],
                         accept_multiple_files=True)

if st.button("Process orders") and files:
    rows, bar = [], st.progress(0.0)
    for i, f in enumerate(files):
        with st.spinner(f"Listening to {f.name}"):
            text = transcribe(f, wsize, lang)
            st.session_state.transcripts[f.name] = text
            for o in extract(text, llm):
                o["source"] = f.name
                rows.append(o)
        bar.progress((i + 1) / len(files))
    st.session_state.orders = pd.DataFrame(rows)

df = st.session_state.orders
if not df.empty:
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(1)
    m1, m2, m3 = st.columns(3)
    for col, label, val in [(m1, "Voice notes", len(st.session_state.transcripts)),
                            (m2, "Order lines", len(df)),
                            (m3, "Customers", df["customer"].replace("", pd.NA).nunique())]:
        col.markdown(f'<div class="card"><div class="lbl">{label}</div><div class="stat">{val}</div></div>',
                     unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Order sheet", "Daily prep list", "Transcripts"])
    with t1:
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        st.download_button("Download CSV", edited.to_csv(index=False).encode("utf-8-sig"),
                           "orders.csv", "text/csv")
    with t2:
        prep = (edited.groupby(["delivery_date", "item"], dropna=False)["quantity"]
                .sum().reset_index().sort_values("delivery_date"))
        st.dataframe(prep, use_container_width=True, hide_index=True)
    with t3:
        for name, text in st.session_state.transcripts.items():
            st.markdown(f'<div class="card"><div class="lbl">{name}</div>{text}</div>',
                        unsafe_allow_html=True)
else:
    st.info("Upload one or more voice notes and press Process orders.")
