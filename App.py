"""
Air Rifle Posture Assistant  -  Streamlit web app.

Run locally :  streamlit run app.py
Deploy      :  push the folder to GitHub -> share.streamlit.io (see README steps)
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

import Posture_core as pc

BASE = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Hard-coded reference images
#   1st path = copy inside the project (works on the web)
#   2nd path = your original D:\ file (works only on your own PC)
# --------------------------------------------------------------------------- #
REFERENCE_FILES = {
    "Front": [BASE / "assets" / "front.png",
              Path(r"D:\Portfolio\New folder\Screenshot 2026-11-21 035935.png")],
    "Side": [BASE / "assets" / "side.png",
             Path(r"D:\Portfolio\New folder\Screenshot 2026-11-21 040425.png")],
    "Back": [BASE / "assets" / "back.png",
             Path(r"D:\Portfolio\New folder\Screenshot 2026-11-21 040205.png")],
}

VIEW_NOTES = {
    "Front": "Camera in front of the shooter, full body in frame.",
    "Side": "Camera at 90 degrees to the target line, at chest height, full body in frame.",
    "Back": "Camera behind the shooter, full body in frame.",
}

st.set_page_config(page_title="Air Rifle Posture Assistant", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, label, p { font-family: 'Barlow', system-ui, sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; max-width: 1400px; }
.app-title { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:2.3rem;
             letter-spacing:.2px; margin:0; line-height:1.05; color:#F1F4F8; }
.app-sub { color:#93A1B5; margin:.25rem 0 1.1rem 0; font-size:1rem; }
.rule { height:3px; width:64px; background:#E8B04B; border-radius:2px; margin:.55rem 0 .7rem 0; }
.card { background:#161D28; border:1px solid #243041; border-radius:14px; padding:14px 16px; }
.card h4 { font-family:'Barlow Condensed',sans-serif; font-weight:600; font-size:1.15rem;
           margin:0 0 .6rem 0; color:#F1F4F8; letter-spacing:.3px; }
.refimg { width:100%; aspect-ratio:3/4; object-fit:contain; background:#14202B; border-radius:10px; display:block; }
.row { display:flex; align-items:center; justify-content:space-between; padding:9px 0;
       border-bottom:1px solid #223046; }
.row:last-child { border-bottom:none; }
.lbl { color:#C6D0DD; font-weight:500; }
.val { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.35rem; min-width:52px; text-align:right; }
.tgt { color:#7F8FA6; font-size:.85rem; min-width:76px; text-align:right; }
.dot { width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:10px; }
.ok { background:#22C55E; } .bad { background:#EF4444; } .na { background:#9CA3AF; }
.c-ok { color:#4ADE80; } .c-bad { color:#F87171; } .c-na { color:#9CA3AF; }
.score { font-family:'Barlow Condensed',sans-serif; font-size:2.1rem; font-weight:700; color:#F1F4F8; }
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.8rem; border:1px solid #2B3A52; color:#9FB0C6; }
.hint { color:#FBBF24; font-weight:500; margin-top:.5rem; }
div[data-testid="stRadio"] label p { font-weight:600; }
video { border-radius:14px; background:#0B1018; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Load references once per server
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Analysing the reference photos...")
def get_refs():
    refs, errors = {}, {}
    for view, paths in REFERENCE_FILES.items():
        try:
            refs[view] = pc.load_reference(view, paths)
        except Exception as e:                      # keep the app alive, show the error
            errors[view] = str(e)
    return refs, errors


REFS, REF_ERRORS = get_refs()


@st.cache_data(show_spinner=False)
def reference_card_png(view: str, flip: bool) -> str:
    rgb, _ = pc.render_reference_card(REFS[view], flip, cutout=False)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def ice_servers():
    """STUN by default.  Add TWILIO_* secrets for a TURN server (needed on some networks)."""
    try:
        from twilio.rest import Client
        c = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
        return c.tokens.create().ice_servers
    except Exception:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


# --------------------------------------------------------------------------- #
# Video processor (runs in a worker thread)
# --------------------------------------------------------------------------- #
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine = pc.PostureEngine(REFS)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        out = self.engine.process(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(out, format="bgr24")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Settings")
    hand = st.radio("Shooter", ["Right-handed", "Left-handed"], horizontal=True,
                    help="Left-handed mirrors the reference photo.")
    tol = st.slider("Tolerance (± degrees)", 3, 25, 10,
                    help="How close each angle must be to the reference to turn green.")
    opacity = st.slider("Reference opacity", 10, 85, 45, step=5)
    st.markdown("#### Overlay")
    autofit = st.toggle("Auto-fit reference to my body", True,
                        help="Scales and moves the reference so it follows your size and position. "
                             "Only posture differences remain.")
    cutout = st.toggle("Isolate the shooter (remove photo background)", True)
    show_ref = st.toggle("Show reference skeleton", False)
    show_live = st.toggle("Show my skeleton", True)
    st.markdown("#### Camera")
    mirror_cam = st.toggle("Mirror camera image", False)
    mirror_ref = st.toggle("Mirror reference photo", False,
                           help="Use if you face the opposite direction from the reference.")
    speed = st.selectbox("Tracking", ["Accurate", "Fast (weaker PC)"], index=0)
    with st.expander("How it works"):
        st.write(
            "MediaPipe Pose finds 33 body landmarks. Five angles are measured on you and on the "
            "reference photo. An angle turns green when it is within the tolerance of the "
            "reference value, otherwise red. Grey means that joint is not visible.")

cfg = pc.Config(view="Side", left_handed=hand.startswith("Left"), mirror_camera=mirror_cam,
                mirror_ref=mirror_ref, opacity=opacity, tolerance=float(tol), cutout=cutout,
                autofit=autofit, show_ref_skeleton=show_ref, show_live_skeleton=show_live,
                complexity=1 if speed == "Accurate" else 0)

# --------------------------------------------------------------------------- #
# Header + view selector
# --------------------------------------------------------------------------- #
st.markdown('<p class="app-title">Air Rifle Posture Assistant</p><div class="rule"></div>'
            '<p class="app-sub">Match your standing position to the reference. '
            'Every joint angle updates live.</p>', unsafe_allow_html=True)

view = st.radio("View", ["Front", "Side", "Back"], index=1, horizontal=True,
                label_visibility="collapsed")
cfg.view = view
st.caption(VIEW_NOTES[view])

if REF_ERRORS:
    for v, msg in REF_ERRORS.items():
        st.error(msg)
if view not in REFS:
    st.stop()

flip = cfg.left_handed != cfg.mirror_ref

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
left, right = st.columns([2.6, 1.15], gap="large")

with left:
    ctx = webrtc_streamer(
        key="posture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": ice_servers()}),
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720},
                                            "frameRate": {"ideal": 30}}, "audio": False},
        video_processor_factory=PostureProcessor,
        async_processing=True,
        video_html_attrs={"style": {"width": "100%", "borderRadius": "14px"},
                          "controls": False, "autoPlay": True, "muted": True},
    )
    if not ctx.state.playing:
        st.info("Press START, allow camera access, then step back until your whole body is visible.")

with right:
    png = reference_card_png(view, flip)
    st.markdown(
        f'<div class="card"><h4>Reference · {view}</h4>'
        f'<img class="refimg" src="data:image/png;base64,{png}"/></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    panel = st.empty()


def render_panel(res: dict | None):
    rows = (res or {}).get("rows", [])
    if not rows:
        panel.markdown('<div class="card"><h4>Live angles</h4>'
                       '<span class="c-na">Waiting for the camera…</span></div>', unsafe_allow_html=True)
        return
    html = [f'<div class="card"><h4>Live angles</h4>'
            f'<div class="score">{res["ok"]}/{res["total"]} '
            f'<span class="pill">±{int(cfg.tolerance)}°</span> '
            f'<span class="pill">{res["fps"]:.0f} fps</span></div>']
    for r in rows:
        live = "--" if r["live"] is None else pc.fmt(r["kind"], r["live"]) + "°"
        tgt = f'ref {r["target_txt"]}°' if r["target_txt"] else ""
        html.append(
            f'<div class="row"><span class="lbl"><span class="dot {r["status"]}"></span>{r["label"]}</span>'
            f'<span class="tgt">{tgt}</span><span class="val c-{r["status"]}">{live}</span></div>')
    if res.get("hint"):
        html.append(f'<div class="hint">{res["hint"]}</div>')
    html.append("</div>")
    panel.markdown("".join(html), unsafe_allow_html=True)


render_panel(None)

# push settings into the running processor and keep the table live
while ctx.state.playing:
    proc = ctx.video_processor
    if proc is not None:
        proc.engine.set_config(cfg)
        render_panel(proc.engine.get_result())
    time.sleep(0.25)