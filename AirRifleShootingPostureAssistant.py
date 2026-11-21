"""
Shooting Posture Assistant (Streamlit + streamlit-webrtc + MediaPipe + OpenCV)

Run locally:   streamlit run app.py
Reference image + live camera -> per-angle red/green feedback, in real time.
"""
from collections import namedtuple

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import (RTCConfiguration, VideoProcessorBase,
                              WebRtcMode, webrtc_streamer)

# --------------------------------------------------------------------------
# Configuration: edit DEFS to match the coach's angles
# --------------------------------------------------------------------------
# MediaPipe landmark indices as (left, right)
IDX = {
    "ear": (7, 8), "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28),
}

# id, label, points, kind ("joint" = angle at the middle point,
#                          "lean"  = angle of segment p0->p1 from vertical)
DEFS = [
    ("elbow",    "Elbow",      ("shoulder", "elbow", "wrist"), "joint", True),
    ("shoulder", "Shoulder",   ("hip", "shoulder", "elbow"),   "joint", True),
    ("hip",      "Hip",        ("shoulder", "hip", "knee"),    "joint", True),
    ("knee",     "Knee",       ("hip", "knee", "ankle"),       "joint", True),
    ("lean",     "Trunk lean", ("hip", "shoulder"),            "lean",  True),
    ("neck",     "Neck",       ("ear", "shoulder", "hip"),     "joint", False),
]
BONES = [("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"),
         ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")]

MIN_VIS = 0.5     # ignore landmarks the model is not confident about
HYST = 2.0        # degrees of hysteresis so colours don't flicker

GREEN = (89, 170, 31)    # BGR
RED = (62, 56, 224)
GREY = (102, 102, 102)
WHITE = (255, 255, 255)
BLUE = (255, 170, 80)

P = namedtuple("P", "x y visibility")
mp_pose = mp.solutions.pose


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def pt(lm, name, side):
    return lm[IDX[name][side]]


def vis_ok(p):
    return p.visibility >= MIN_VIS


def pick_side(lm):
    """Choose the body side that faces the camera (more visible landmarks)."""
    score = lambda s: sum(lm[i + s].visibility for i in (11, 23, 25))
    return 1 if score(1) > score(0) else 0


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


def compute_angles(lm, W, H, side):
    out = {}
    for did, _label, names, kind, _on in DEFS:
        pts = [pt(lm, n, side) for n in names]
        if not all(vis_ok(p) for p in pts):
            out[did] = None
        elif kind == "lean":
            out[did] = lean_angle(pts[0], pts[1], W, H)
        else:
            out[did] = joint_angle(pts[0], pts[1], pts[2], W, H)
    return out


def detect_static(bgr):
    """Run pose detection on a still image. Returns list of P or None."""
    with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                      min_detection_confidence=0.5) as pose:
        res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None
    return [P(l.x, l.y, l.visibility) for l in res.pose_landmarks.landmark]


def draw_label(img, text, x, y, color, scale):
    H, W = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(max(0, min(x, W - tw - 12)))
    y = int(max(0, min(y, H - th - 12)))
    cv2.rectangle(img, (x, y), (x + tw + 10, y + th + 10), color, -1)
    cv2.putText(img, text, (x + 5, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                scale, WHITE, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Video processor: runs on every frame from the browser camera
# --------------------------------------------------------------------------
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        self.cfg = None            # set from the Streamlit script on each rerun
        self.state = {}            # hysteresis memory per angle
        self.last_frame = None     # used by "use current frame as reference"
        self._ref_cache = (None, None)

    # ---- ghost reference image blended over the live frame
    def _draw_ghost(self, img, cfg):
        ref = cfg["ref_img"]
        if ref is None:
            return None
        H, W = img.shape[:2]
        rh, rw = ref.shape[:2]
        s = min(W / rw, H / rh)
        nw, nh = int(rw * s), int(rh * s)
        x0, y0 = (W - nw) // 2, (H - nh) // 2
        key = (W, H, cfg["flip_ref"], id(ref))
        if self._ref_cache[0] != key:
            r = cv2.resize(ref, (nw, nh))
            if cfg["flip_ref"]:
                r = cv2.flip(r, 1)
            self._ref_cache = (key, r)
        r = self._ref_cache[1]
        a = cfg["opacity"]
        roi = img[y0:y0 + nh, x0:x0 + nw]
        img[y0:y0 + nh, x0:x0 + nw] = cv2.addWeighted(roi, 1 - a, r, a, 0)
        return x0, y0, nw, nh

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = self.cfg
        if cfg is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self.last_frame = img.copy()
        if cfg["mirror"]:
            img = cv2.flip(img, 1)
        H, W = img.shape[:2]
        scale = max(0.5, W / 1280 * 0.7)

        # detect on the clean frame, then draw overlays
        res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        rect = self._draw_ghost(img, cfg)

        # reference skeleton
        if rect and cfg["show_ref_skel"] and cfg["ref_lm"]:
            x0, y0, nw, nh = rect
            rs = cfg["ref_side"]

            def rp(name):
                p = pt(cfg["ref_lm"], name, rs)
                px = (1 - p.x) if cfg["flip_ref"] else p.x
                return int(x0 + px * nw), int(y0 + p.y * nh)

            for a, b in BONES:
                cv2.line(img, rp(a), rp(b), BLUE, 3, cv2.LINE_AA)

        banner, banner_color = "No person detected", GREY
        if res.pose_landmarks:
            lm = [P(l.x, l.y, l.visibility) for l in res.pose_landmarks.landmark]
            side = pick_side(lm) if cfg["side"] == "auto" else int(cfg["side"])

            def lp(name):
                p = pt(lm, name, side)
                return int(p.x * W), int(p.y * H), vis_ok(p)

            for a, b in BONES:
                A, B = lp(a), lp(b)
                if A[2] and B[2]:
                    cv2.line(img, A[:2], B[:2], WHITE, 3, cv2.LINE_AA)

            cur = compute_angles(lm, W, H, side)
            ref_angles = cfg["ref_angles"]
            tol = cfg["tol"]
            stack, ok_n, counted = {}, 0, 0

            for did, label, names, kind, _on in DEFS:
                if did not in cfg["enabled"]:
                    continue
                c = cur[did]
                r = ref_angles.get(did) if ref_angles else None
                color = GREY
                if c is None:
                    text = f"{label} ?"
                elif r is None:
                    text = f"{label} {c:.0f} deg"
                else:
                    diff = abs(c - r)
                    prev = self.state.get(did, "bad")
                    if prev == "ok":
                        self.state[did] = "bad" if diff > tol + HYST else "ok"
                    else:
                        self.state[did] = "ok" if diff <= tol else "bad"
                    color = GREEN if self.state[did] == "ok" else RED
                    counted += 1
                    ok_n += self.state[did] == "ok"
                    text = f"{label} {c:.0f} deg (ref {r:.0f})"

                if kind == "lean":
                    A, B = lp(names[0]), lp(names[1])
                    ax, ay, key = (A[0] + B[0]) // 2, (A[1] + B[1]) // 2, "lean"
                else:
                    ax, ay, _ = lp(names[1])
                    key = names[1]
                k = stack[key] = stack.get(key, -1) + 1
                cv2.circle(img, (ax, ay), 9, color, -1)
                cv2.circle(img, (ax, ay), 9, WHITE, 2)
                draw_label(img, text, ax + 14, ay - 12 + k * int(34 * scale), color, scale)

            if ref_angles is None:
                banner, banner_color = "Load a reference image", GREY
            elif counted == 0:
                banner, banner_color = "Angles not visible", GREY
            elif ok_n == counted:
                banner, banner_color = f"Matched {ok_n}/{counted} - hold", GREEN
                cv2.rectangle(img, (4, 4), (W - 5, H - 5), GREEN, 8)
            else:
                banner, banner_color = f"{ok_n}/{counted} in range", RED
                cv2.rectangle(img, (4, 4), (W - 5, H - 5), RED, 8)

        draw_label(img, banner, 12, 12, banner_color, scale * 1.1)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# Streamlit page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Shooting Posture Assistant", layout="wide")
st.title("Shooting Posture Assistant")

if "ref_img" not in st.session_state:
    st.session_state.update(ref_img=None, ref_lm=None, ref_key=None)

with st.sidebar:
    st.header("1. Reference")
    up = st.file_uploader("Photo of the correct stance (side view)",
                          type=["png", "jpg", "jpeg"])
    if up is not None:
        key = (up.name, up.size)
        if st.session_state.ref_key != key:
            bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            lm = detect_static(bgr)
            if lm is None:
                st.error("No person detected in that image.")
            else:
                st.session_state.update(ref_img=bgr, ref_lm=lm, ref_key=key)

    st.header("2. Settings")
    tol = st.slider("Tolerance (± degrees)", 3, 25, 10)
    opacity = st.slider("Ghost opacity", 0, 100, 45) / 100
    side_choice = st.selectbox("Body side facing the camera", ["auto", "left", "right"])
    side = {"auto": "auto", "left": 0, "right": 1}[side_choice]
    enabled = [d[0] for d in DEFS if st.checkbox(d[1], value=d[4], key=f"chk_{d[0]}")]
    mirror = st.checkbox("Mirror live video", value=False)
    flip_ref = st.checkbox("Flip reference horizontally", value=False)
    show_ref_skel = st.checkbox("Show reference skeleton", value=True)

# reference angles (recomputed when the side setting changes)
ref_lm, ref_img = st.session_state.ref_lm, st.session_state.ref_img
ref_angles, ref_side = None, 0
if ref_lm is not None:
    ref_side = pick_side(ref_lm) if side == "auto" else int(side)
    h, w = ref_img.shape[:2]
    ref_angles = compute_angles(ref_lm, w, h, ref_side)

st.caption("Press START below to turn the camera on, STOP to turn it off. "
           "Angles are compared by value, so you don't have to match the ghost's "
           "exact position or size.")

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

if ctx.video_processor:
    ctx.video_processor.cfg = dict(
        ref_img=ref_img, ref_lm=ref_lm, ref_angles=ref_angles, ref_side=ref_side,
        tol=tol, opacity=opacity, side=side, enabled=enabled,
        mirror=mirror, flip_ref=flip_ref, show_ref_skel=show_ref_skel,
    )
    if st.button("Use current camera frame as the reference"):
        frame = ctx.video_processor.last_frame
        lm = detect_static(frame) if frame is not None else None
        if lm is None:
            st.error("No person detected in the current frame.")
        else:
            st.session_state.update(ref_img=frame, ref_lm=lm, ref_key=("camera", id(frame)))
            st.rerun()
elif ref_img is None:
    st.info("Upload a reference image in the sidebar, then press START.")
    