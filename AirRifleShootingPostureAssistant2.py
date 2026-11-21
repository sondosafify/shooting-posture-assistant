"""
Shooting Posture Assistant  -  Side / Front / Back views
Streamlit + streamlit-webrtc + MediaPipe + OpenCV

Run locally:   streamlit run app.py
"""
from collections import namedtuple
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import (RTCConfiguration, VideoProcessorBase,
                              WebRtcMode, webrtc_streamer)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Hard-coded reference images.
# Each view lists paths to try in order. The D:\ paths work on your own PC.
# For a hosted link, copy the three images into an "assets" folder next to
# app.py (named side.png, front.png, back.png) and upload them with the code.
# --------------------------------------------------------------------------
REFERENCE_PATHS = {
    "Side": [r"D:\Portfolio\New folder\Screenshot 2026-11-21 040425.png", "assets/side.png"],
    "Front": [r"D:\Portfolio\New folder\Screenshot 2026-11-21 035935.png", "assets/front.png"],
    "Back": [r"D:\Portfolio\New folder\Screenshot 2026-11-21 040205.png", "assets/back.png"],
}
VIEWS = list(REFERENCE_PATHS)

TIPS = {
    "Side": "Place the camera at chest height, perpendicular to the firing line, with "
            "the whole body in frame and the shoulder-side of the rifle facing the camera.",
    "Front": "Place the camera directly in front of the shooter at chest height, far "
             "enough back to show head to feet.",
    "Back": "Place the camera directly behind the shooter at chest height, far enough "
            "back to show head to feet.",
}

# --------------------------------------------------------------------------
# Landmarks and angle definitions (edit these to match the coach's angles)
# --------------------------------------------------------------------------
# MediaPipe landmark indices as (left, right) of the shooter
IDX = {
    "ear": (7, 8), "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28),
}
# A point is (name, who): who = "near" (side facing camera), "L", "R" or "mid".
# kind: "joint" = bend at the middle point (3 points)
#       "lean"  = tilt of segment p0->p1 from vertical (2 points)
#       "tilt"  = tilt of segment p0->p1 from horizontal (2 points)
DEFS = {
    "Side": [
        dict(id="elbow", label="Elbow", kind="joint", on=True,
             pts=[("shoulder", "near"), ("elbow", "near"), ("wrist", "near")]),
        dict(id="shoulder", label="Shoulder", kind="joint", on=True,
             pts=[("hip", "near"), ("shoulder", "near"), ("elbow", "near")]),
        dict(id="hip", label="Hip", kind="joint", on=True,
             pts=[("shoulder", "near"), ("hip", "near"), ("knee", "near")]),
        dict(id="knee", label="Knee", kind="joint", on=True,
             pts=[("hip", "near"), ("knee", "near"), ("ankle", "near")]),
        dict(id="lean", label="Trunk lean", kind="lean", on=True,
             pts=[("hip", "near"), ("shoulder", "near")]),
        dict(id="neck", label="Neck", kind="joint", on=False,
             pts=[("ear", "near"), ("shoulder", "near"), ("hip", "near")]),
    ],
}
_FRONT_BACK = [
    dict(id="shoulder_tilt", label="Shoulder tilt", kind="tilt", on=True,
         pts=[("shoulder", "L"), ("shoulder", "R")]),
    dict(id="hip_tilt", label="Hip tilt", kind="tilt", on=True,
         pts=[("hip", "L"), ("hip", "R")]),
    dict(id="trunk_lean", label="Trunk lean", kind="lean", on=True,
         pts=[("hip", "mid"), ("shoulder", "mid")]),
    dict(id="l_knee", label="L knee", kind="joint", on=True,
         pts=[("hip", "L"), ("knee", "L"), ("ankle", "L")]),
    dict(id="r_knee", label="R knee", kind="joint", on=True,
         pts=[("hip", "R"), ("knee", "R"), ("ankle", "R")]),
    dict(id="head_tilt", label="Head tilt", kind="tilt", on=False,
         pts=[("ear", "L"), ("ear", "R")]),
    dict(id="l_elbow", label="L elbow", kind="joint", on=False,
         pts=[("shoulder", "L"), ("elbow", "L"), ("wrist", "L")]),
    dict(id="r_elbow", label="R elbow", kind="joint", on=False,
         pts=[("shoulder", "R"), ("elbow", "R"), ("wrist", "R")]),
]
DEFS["Front"] = _FRONT_BACK
DEFS["Back"] = _FRONT_BACK

_CHAIN = [("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"),
          ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")]
BONES = {
    "Side": [((a, "near"), (b, "near")) for a, b in _CHAIN],
    "Front": ([((a, w), (b, w)) for w in ("L", "R") for a, b in _CHAIN]
              + [(("shoulder", "L"), ("shoulder", "R")), (("hip", "L"), ("hip", "R"))]),
}
BONES["Back"] = BONES["Front"]

MIN_VIS = 0.5     # ignore landmarks the model is not confident about
HYST = 2.0        # degrees of hysteresis so colours don't flicker

GREEN = (89, 170, 31)    # BGR
RED = (62, 56, 224)
GREY = (102, 102, 102)
WHITE = (255, 255, 255)
BLUE = (255, 170, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX

P = namedtuple("P", "x y visibility")
mp_pose = mp.solutions.pose


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def vis_ok(p):
    return p.visibility >= MIN_VIS


def pick_side(lm):
    """Body side that faces the camera (more confidently visible landmarks)."""
    score = lambda s: sum(lm[i + s].visibility for i in (11, 23, 25))
    return 1 if score(1) > score(0) else 0


def resolve(lm, name, who, side):
    if who == "mid":
        a, b = lm[IDX[name][0]], lm[IDX[name][1]]
        return P((a.x + b.x) / 2, (a.y + b.y) / 2, min(a.visibility, b.visibility))
    s = side if who == "near" else (0 if who == "L" else 1)
    return lm[IDX[name][s]]


def joint_angle(a, b, c, W, H):
    v1 = np.array([(a.x - b.x) * W, (a.y - b.y) * H])
    v2 = np.array([(c.x - b.x) * W, (c.y - b.y) * H])
    m = np.linalg.norm(v1) * np.linalg.norm(v2)
    if m == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / m, -1, 1))))


def lean_angle(a, b, W, H):
    """Angle of segment a->b from vertical (0 = upright)."""
    return float(np.degrees(np.arctan2(abs((b.x - a.x) * W), abs((b.y - a.y) * H))))


def tilt_angle(a, b, W, H):
    """Angle of segment a->b from horizontal (0 = level)."""
    return float(np.degrees(np.arctan2(abs((b.y - a.y) * H), abs((b.x - a.x) * W))))


def compute_angles(lm, W, H, side, defs):
    out = {}
    for d in defs:
        pts = [resolve(lm, n, w, side) for n, w in d["pts"]]
        if not all(vis_ok(p) for p in pts):
            out[d["id"]] = None
        elif d["kind"] == "joint":
            out[d["id"]] = joint_angle(pts[0], pts[1], pts[2], W, H)
        elif d["kind"] == "lean":
            out[d["id"]] = lean_angle(pts[0], pts[1], W, H)
        else:
            out[d["id"]] = tilt_angle(pts[0], pts[1], W, H)
    return out


def detect_static(bgr):
    with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                      min_detection_confidence=0.5) as pose:
        res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None
    return [P(l.x, l.y, l.visibility) for l in res.pose_landmarks.landmark]


@st.cache_resource(show_spinner=False)
def load_reference(view):
    """Load and analyse the hard-coded reference image for a view."""
    tried = []
    for p in REFERENCE_PATHS[view]:
        path = Path(p)
        if not path.is_absolute():
            path = BASE_DIR / p
        tried.append(str(path))
        if not path.exists():
            continue
        bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if max(h, w) > 1600:                      # keep processing fast
            f = 1600 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        return dict(img=bgr, lm=detect_static(bgr), path=str(path), tried=tried)
    return dict(img=None, lm=None, path=None, tried=tried)


# --------------------------------------------------------------------------
# Consistent ghost size: scale each reference so the PERSON (not the photo)
# fills the same share of the frame, centred, with nothing of the person cut.
# --------------------------------------------------------------------------
def person_bbox(lm, w, h, flip):
    pts = [(p.x * w, p.y * h) for p in lm if p.visibility > 0.3]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = max(0, min(xs)), min(w, max(xs))
    y0, y1 = max(0, min(ys)), min(h, max(ys))
    px, py = 0.04 * (x1 - x0), 0.04 * (y1 - y0)
    x0, x1, y0, y1 = max(0, x0 - px), min(w, x1 + px), max(0, y0 - py), min(h, y1 + py)
    if flip:
        x0, x1 = w - x1, w - x0
    return x0, y0, x1, y1


def ghost_layout(ref, lm, W, H, flip, size_frac):
    """Return (canvas, mask, (nw, nh, ox, oy)) with the person centred and scaled."""
    rh, rw = ref.shape[:2]
    x0, y0, x1, y1 = person_bbox(lm, rw, rh, flip)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    s = min(size_frac * H / bh, 0.92 * W / bw)
    tx = W / 2 - (x0 + x1) / 2 * s
    ty = H / 2 - (y0 + y1) / 2 * s
    src = cv2.flip(ref, 1) if flip else ref
    nw, nh = max(1, round(rw * s)), max(1, round(rh * s))
    resized = cv2.resize(src, (nw, nh),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), bool)
    ox, oy = int(round(tx)), int(round(ty))
    xa, ya, xb, yb = max(0, ox), max(0, oy), min(W, ox + nw), min(H, oy + nh)
    if xb > xa and yb > ya:
        canvas[ya:yb, xa:xb] = resized[ya - oy:yb - oy, xa - ox:xb - ox]
        mask[ya:yb, xa:xb] = True
    return canvas, mask, (nw, nh, ox, oy)


def draw_label(img, text, x, y, color, scale, align="left"):
    H, W = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    bw = tw + 12
    if align == "right":              # box's right edge sits at x
        x -= bw
    x = int(max(0, min(x, W - bw)))
    y = int(max(0, min(y, H - th - 12)))
    cv2.rectangle(img, (x, y), (x + bw, y + th + 10), color, -1)
    cv2.putText(img, text, (x + 6, y + th + 3), FONT, scale, WHITE, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Video processor: runs on every frame from the browser camera
# --------------------------------------------------------------------------
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        self.cfg = None          # set from the Streamlit script on each rerun
        self.view = None
        self.state = {}          # hysteresis memory per angle
        self._ghost_key, self._ghost = None, None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = self.cfg
        if cfg is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        view = cfg["view"]
        if view != self.view:               # new view: forget old colour states
            self.view, self.state = view, {}
        if cfg["mirror"]:
            img = cv2.flip(img, 1)
        H, W = img.shape[:2]
        scale = max(0.5, W / 1280 * 0.7)
        defs = DEFS[view]

        # detect on the clean frame, then draw overlays
        res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # ---- ghost reference
        ref_lm, layout = cfg["ref_lm"], None
        if cfg["ref_img"] is not None and ref_lm is not None:
            key = (view, W, H, cfg["flip_ref"], round(cfg["size_frac"], 2))
            if key != self._ghost_key:
                self._ghost = ghost_layout(cfg["ref_img"], ref_lm, W, H,
                                           cfg["flip_ref"], cfg["size_frac"])
                self._ghost_key = key
            canvas, mask, layout = self._ghost
            a = cfg["opacity"]
            blended = cv2.addWeighted(img, 1 - a, canvas, a, 0)
            img[mask] = blended[mask]

            if cfg["show_ref_skel"]:
                nw, nh, ox, oy = layout
                rs = cfg["ref_side"]

                def rp(spec):
                    p = resolve(ref_lm, spec[0], spec[1], rs)
                    x = (1 - p.x) if cfg["flip_ref"] else p.x
                    return int(x * nw + ox), int(p.y * nh + oy), vis_ok(p)

                for a_, b_ in BONES[view]:
                    A, B = rp(a_), rp(b_)
                    if A[2] and B[2]:
                        cv2.line(img, A[:2], B[:2], BLUE, 3, cv2.LINE_AA)

        banner, banner_color = f"{view} view: no person detected", GREY
        if res.pose_landmarks:
            lm = [P(l.x, l.y, l.visibility) for l in res.pose_landmarks.landmark]
            side = pick_side(lm) if cfg["side"] == "auto" else int(cfg["side"])

            for a_, b_ in BONES[view]:
                pa, pb = resolve(lm, a_[0], a_[1], side), resolve(lm, b_[0], b_[1], side)
                if vis_ok(pa) and vis_ok(pb):
                    cv2.line(img, (int(pa.x * W), int(pa.y * H)),
                             (int(pb.x * W), int(pb.y * H)), WHITE, 3, cv2.LINE_AA)

            cur = compute_angles(lm, W, H, side, defs)
            ref_angles, tol = cfg["ref_angles"], cfg["tol"]
            stack, ok_n, counted = {}, 0, 0

            for d in defs:
                if d["id"] not in cfg["enabled"]:
                    continue
                c = cur[d["id"]]
                r = ref_angles.get(d["id"]) if ref_angles else None
                color = GREY
                if c is None:
                    text = f"{d['label']} ?"
                elif r is None:
                    text = f"{d['label']} {c:.0f}"
                else:
                    diff = abs(c - r)
                    prev = self.state.get(d["id"], "bad")
                    if prev == "ok":
                        self.state[d["id"]] = "bad" if diff > tol + HYST else "ok"
                    else:
                        self.state[d["id"]] = "ok" if diff <= tol else "bad"
                    ok = self.state[d["id"]] == "ok"
                    color = GREEN if ok else RED
                    counted += 1
                    ok_n += ok
                    text = f"{d['label']} {c:.0f} (ref {r:.0f})"

                pts = [resolve(lm, n, w, side) for n, w in d["pts"]]
                if d["kind"] == "joint":
                    ax, ay, key = pts[1].x, pts[1].y, tuple(d["pts"][1])
                else:
                    ax, ay, key = (pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2, d["id"]
                ax, ay = int(ax * W), int(ay * H)
                k = stack[key] = stack.get(key, -1) + 1
                cv2.circle(img, (ax, ay), 9, color, -1)
                cv2.circle(img, (ax, ay), 9, WHITE, 2)
                ly = ay - 12 + k * int(40 * scale)
                # labels go outward from the body centre so neighbours don't collide
                if ax < W / 2:
                    draw_label(img, text, ax - 14, ly, color, scale, align="right")
                else:
                    draw_label(img, text, ax + 14, ly, color, scale)

            if ref_angles is None:
                banner, banner_color = f"{view} view: reference not available", GREY
            elif counted == 0:
                banner, banner_color = f"{view} view: angles not visible", GREY
            elif ok_n == counted:
                banner, banner_color = f"{view} view: matched {ok_n}/{counted} - hold", GREEN
                cv2.rectangle(img, (4, 4), (W - 5, H - 5), GREEN, 8)
            else:
                banner, banner_color = f"{view} view: {ok_n}/{counted} in range", RED
                cv2.rectangle(img, (4, 4), (W - 5, H - 5), RED, 8)

        draw_label(img, banner, 12, 12, banner_color, scale * 1.1)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# Streamlit page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Shooting Posture Assistant", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
#MainMenu, footer {visibility: hidden;}
.app-header {background: linear-gradient(135deg,#0f2027 0%,#203a43 55%,#2c5364 100%);
             padding: 22px 28px; border-radius: 14px; color: #fff; margin-bottom: 18px;}
.app-header h1 {margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: .2px; color:#fff;}
.app-header p {margin: 4px 0 0; opacity: .82; font-size: .95rem;}
.card {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.22);
       border-radius: 12px; padding: 14px 16px; margin-bottom: 14px;}
.card h4 {margin: 0 0 8px; font-size: .78rem; text-transform: uppercase;
          letter-spacing: .08em; opacity: .65;}
.card p {margin: 0; font-size: .9rem; line-height: 1.45;}
.legend {display: flex; gap: 14px; flex-wrap: wrap; font-size: .85rem; margin: 2px 0 12px;}
.legend i {display: inline-block; width: 11px; height: 11px; border-radius: 50%;
           margin-right: 6px; vertical-align: -1px;}
div[role="radiogroup"] {gap: .5rem;}
div[role="radiogroup"] label {background: rgba(128,128,128,.12);
    border: 1px solid rgba(128,128,128,.28); padding: 6px 26px; border-radius: 999px;
    cursor: pointer;}
div[role="radiogroup"] label > div:first-child {display: none;}
div[role="radiogroup"] label:has(input:checked) {background: #2563eb; border-color: #2563eb;}
div[role="radiogroup"] label:has(input:checked) p {color: #fff; font-weight: 600;}
</style>
<div class="app-header">
  <h1>Shooting Posture Assistant</h1>
  <p>Match the ghost reference. Each joint turns green when its angle is within tolerance.</p>
</div>
""", unsafe_allow_html=True)

view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed")
defs = DEFS[view]
ref = load_reference(view)
ref_img, ref_lm = ref["img"], ref["lm"]

with st.sidebar:
    st.subheader("Settings")
    tol = st.slider("Tolerance (± degrees)", 3, 25, 10)
    opacity = st.slider("Ghost opacity", 0, 100, 45) / 100
    size_frac = st.slider("Ghost size (% of frame height)", 50, 95, 85) / 100
    if view == "Side":
        side_choice = st.selectbox("Body side facing the camera", ["auto", "left", "right"])
        side = {"auto": "auto", "left": 0, "right": 1}[side_choice]
    else:
        side = "auto"
    st.markdown("**Angles to compare**")
    enabled = [d["id"] for d in defs
               if st.checkbox(d["label"], value=d["on"], key=f"chk_{view}_{d['id']}")]
    with st.expander("Advanced"):
        mirror = st.checkbox("Mirror live video", value=False)
        flip_ref = st.checkbox("Flip reference horizontally", value=False)
        show_ref_skel = st.checkbox("Show reference skeleton", value=True)
    st.caption("All angles in degrees. L / R = the shooter's left / right.")

ref_angles, ref_side = None, 0
if ref_lm is not None:
    ref_side = pick_side(ref_lm) if side == "auto" else int(side)
    h, w = ref_img.shape[:2]
    ref_angles = compute_angles(ref_lm, w, h, ref_side, defs)

left, right = st.columns([3, 1.15], gap="large")

with left:
    st.markdown(
        '<div class="legend">'
        '<span><i style="background:#1faa59"></i>Within tolerance</span>'
        '<span><i style="background:#e0383e"></i>Out of range</span>'
        '<span><i style="background:#666"></i>Not visible</span>'
        '<span><i style="background:#50aaff"></i>Reference skeleton</span></div>',
        unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="posture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=PostureProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        async_processing=True,
    )
    st.caption("Press START to turn the camera on and STOP to turn it off. "
               "You can switch views while the camera is running.")

with right:
    st.markdown(f'<div class="card"><h4>Camera setup: {view}</h4><p>{TIPS[view]}</p></div>',
                unsafe_allow_html=True)
    if ref_img is None:
        st.error("Reference image for this view was not found. Tried:\n\n"
                 + "\n".join(f"- {t}" for t in ref["tried"]))
    elif ref_lm is None:
        st.warning("No person could be detected in this reference image.")
    else:
        st.markdown(f'<div class="card"><h4>Reference: {view}</h4></div>', unsafe_allow_html=True)
        canvas, mask, _ = ghost_layout(ref_img, ref_lm, 640, 480, False, 0.9)
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        rows = [{"Angle": d["label"],
                 "Reference (deg)": "n/a" if ref_angles[d["id"]] is None
                 else f"{ref_angles[d['id']]:.0f}"}
                for d in defs if d["id"] in enabled]
        if rows:
            st.table(pd.DataFrame(rows).set_index("Angle"))

if ctx.video_processor:
    ctx.video_processor.cfg = dict(
        view=view, ref_img=ref_img, ref_lm=ref_lm, ref_angles=ref_angles,
        ref_side=ref_side, tol=tol, opacity=opacity, size_frac=size_frac,
        side=side, enabled=enabled, mirror=mirror, flip_ref=flip_ref,
        show_ref_skel=show_ref_skel,
    )