"""
Shooting Posture Assistant  -  Side / Front / Back views
Streamlit + streamlit-webrtc + MediaPipe (heavy BlazePose) + OpenCV

A shaded full-body 3D-style mannequin, posed exactly like the hard-coded
reference photo (always mirrored), is overlaid on the live camera. Each joint
turns green when its angle is within tolerance of the reference, red otherwise.

Run locally:   streamlit run app.py
"""
from collections import namedtuple
from pathlib import Path

import base64

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
# Fixed settings (these used to be sliders)
# --------------------------------------------------------------------------
TOL_DEG = 10.0       # an angle is "in range" when within +/- this many degrees
GHOST_ALPHA = 0.50   # opacity of the reference mannequin over the live video
HYST = 2.0           # degrees of hysteresis so colours don't flicker
MIN_VIS = 0.5        # ignore landmarks the model is not confident about
MODEL_CHOICES = {"Heavy (most accurate)": 2, "Full (faster)": 1}

# --------------------------------------------------------------------------
# Hard-coded reference photos. The D:\ paths work on your own PC; the
# assets/ copies are what a hosted link uses (the server cannot see D:\).
# --------------------------------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_PATHS = {
    "Side": [
        str(BASE_DIR / "assets" / "side.png"),
        str(BASE_DIR / "assetsside.png"),
    ],
    "Front": [
        str(BASE_DIR / "assets" / "front.png"),
        str(BASE_DIR / "assetsfront.png"),
    ],
    "Back": [
        str(BASE_DIR / "assets" / "back.png"),
        str(BASE_DIR / "assetsback.png"),
    ],
}
VIEWS = list(REFERENCE_PATHS)

TIPS = {
    "Side": "Camera at chest height, perpendicular to the firing line, whole body in "
            "frame from head to feet.",
    "Front": "Camera directly in front of the shooter at chest height, far enough "
             "back to show head to feet.",
    "Back": "Camera directly behind the shooter at chest height, far enough back to "
            "show head to feet.",
}

# --------------------------------------------------------------------------
# Landmarks and angle definitions (edit these to match the coach's angles)
# --------------------------------------------------------------------------
# MediaPipe landmark indices as (left chain, right chain)
IDX = {
    "ear": (7, 8), "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "index": (19, 20), "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28),
    "heel": (29, 30), "foot": (31, 32),
}
# A point is (name, who): who = "near" (the body side facing the camera, side view),
# "L" / "R" (left / right half of the SCREEN, front and back views) or "mid".
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
        dict(id="neck", label="Neck", kind="joint", on=True,
             pts=[("ear", "near"), ("shoulder", "near"), ("hip", "near")]),
        dict(id="wrist", label="Wrist", kind="joint", on=False,
             pts=[("elbow", "near"), ("wrist", "near"), ("index", "near")]),
        dict(id="ankle", label="Ankle", kind="joint", on=False,
             pts=[("knee", "near"), ("ankle", "near"), ("foot", "near")]),
    ],
}
_FRONT_BACK = [
    dict(id="shoulder_tilt", label="Shoulder tilt", kind="tilt", on=True,
         pts=[("shoulder", "L"), ("shoulder", "R")]),
    dict(id="hip_tilt", label="Hip tilt", kind="tilt", on=True,
         pts=[("hip", "L"), ("hip", "R")]),
    dict(id="trunk_lean", label="Trunk lean", kind="lean", on=True,
         pts=[("hip", "mid"), ("shoulder", "mid")]),
    dict(id="head_tilt", label="Head tilt", kind="tilt", on=True,
         pts=[("ear", "L"), ("ear", "R")]),
    dict(id="l_elbow", label="Elbow L", kind="joint", on=True,
         pts=[("shoulder", "L"), ("elbow", "L"), ("wrist", "L")]),
    dict(id="r_elbow", label="Elbow R", kind="joint", on=True,
         pts=[("shoulder", "R"), ("elbow", "R"), ("wrist", "R")]),
    dict(id="l_knee", label="Knee L", kind="joint", on=True,
         pts=[("hip", "L"), ("knee", "L"), ("ankle", "L")]),
    dict(id="r_knee", label="Knee R", kind="joint", on=True,
         pts=[("hip", "R"), ("knee", "R"), ("ankle", "R")]),
    dict(id="l_wrist", label="Wrist L", kind="joint", on=False,
         pts=[("elbow", "L"), ("wrist", "L"), ("index", "L")]),
    dict(id="r_wrist", label="Wrist R", kind="joint", on=False,
         pts=[("elbow", "R"), ("wrist", "R"), ("index", "R")]),
    dict(id="l_ankle", label="Ankle L", kind="joint", on=False,
         pts=[("knee", "L"), ("ankle", "L"), ("foot", "L")]),
    dict(id="r_ankle", label="Ankle R", kind="joint", on=False,
         pts=[("knee", "R"), ("ankle", "R"), ("foot", "R")]),
]
DEFS["Front"] = _FRONT_BACK
DEFS["Back"] = _FRONT_BACK

_CHAIN = [("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"),
          ("wrist", "index"), ("shoulder", "hip"), ("hip", "knee"),
          ("knee", "ankle"), ("ankle", "foot")]
BONES = {
    "Side": [((a, "near"), (b, "near")) for a, b in _CHAIN],
    "Front": ([((a, w), (b, w)) for w in ("L", "R") for a, b in _CHAIN]
              + [(("shoulder", "L"), ("shoulder", "R")), (("hip", "L"), ("hip", "R"))]),
}
BONES["Back"] = BONES["Front"]

GREEN = (89, 170, 31)    # BGR
RED = (62, 56, 224)
GREY = (102, 102, 102)
WHITE = (255, 255, 255)
BLUE = (255, 170, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX

P = namedtuple("P", "x y z visibility")
mp_pose = mp.solutions.pose


# --------------------------------------------------------------------------
# Pose model (heavy by default, falls back to the bundled Full model)
# --------------------------------------------------------------------------
def make_pose(complexity, static=False):
    """Create a MediaPipe Pose. The heavy model is downloaded on first use; if that
    is impossible (no internet) the bundled Full model is used instead."""
    tried = [complexity] + ([1] if complexity != 1 else [])
    last = None
    for c in tried:
        try:
            pose = mp_pose.Pose(static_image_mode=static, model_complexity=c,
                                smooth_landmarks=not static,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
            return pose, c
        except Exception as e:          # download failed, etc.
            last = e
    raise RuntimeError(f"Could not start the pose model: {last}")


def to_landmarks(res):
    return [P(l.x, l.y, l.z, l.visibility) for l in res.pose_landmarks.landmark]


def _pose_quality(lm):
    """How plausible a full-body detection is: confident key joints + a tall skeleton."""
    key = [11, 12, 23, 24, 25, 26, 27, 28]
    vis = float(np.mean([lm[i].visibility for i in key]))
    span = max(lm[27].y, lm[28].y) - min(lm[0].y, lm[11].y, lm[12].y)
    return vis + min(max(span, 0.0), 1.0)


def detect_static(bgr):
    """Find the pose in a still photo. The model sometimes locks onto only the upper
    body when the person fills the picture, so it is also tried with grey padding
    around the photo; the most plausible full-body result is kept. Landmarks are
    returned in the ORIGINAL photo's normalised coordinates."""
    pose, used = make_pose(2, static=True)
    h, w = bgr.shape[:2]
    best, best_score = None, -1.0
    try:
        for frac in (0.0, 0.15, 0.30, 0.45):
            px, py = int(w * frac), int(h * frac)
            canvas = (cv2.copyMakeBorder(bgr, py, py, px, px, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114)) if frac else bgr)
            Hp, Wp = canvas.shape[:2]
            res = pose.process(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                continue
            lm = [P((l.x * Wp - px) / w, (l.y * Hp - py) / h, l.z * Wp / w, l.visibility)
                  for l in res.pose_landmarks.landmark]
            score = _pose_quality(lm)
            if score > best_score:
                best, best_score = lm, score
    finally:
        pose.close()
    return best, used


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def vis_ok(p):
    return p.visibility >= MIN_VIS


def pick_side(lm):
    """Body chain that faces the camera (more confidently visible landmarks)."""
    score = lambda s: sum(lm[i + s].visibility for i in (11, 23, 25))
    return 1 if score(1) > score(0) else 0


def chain_order(lm, flip=False):
    """(chain on the screen-left, chain on the screen-right) as displayed.
    flip=True gives the order after the picture is mirrored horizontally."""
    x0 = np.mean([lm[11].x, lm[23].x])
    x1 = np.mean([lm[12].x, lm[24].x])
    if flip:
        x0, x1 = 1 - x0, 1 - x1
    return (0, 1) if x0 <= x1 else (1, 0)


def resolve(lm, name, who, side, order):
    if who == "mid":
        a, b = lm[IDX[name][0]], lm[IDX[name][1]]
        return P((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2,
                 min(a.visibility, b.visibility))
    s = side if who == "near" else (order[0] if who == "L" else order[1])
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


def compute_angles(lm, W, H, side, order, defs):
    out = {}
    for d in defs:
        pts = [resolve(lm, n, w, side, order) for n, w in d["pts"]]
        if not all(vis_ok(p) for p in pts):
            out[d["id"]] = None
        elif d["kind"] == "joint":
            out[d["id"]] = joint_angle(pts[0], pts[1], pts[2], W, H)
        elif d["kind"] == "lean":
            out[d["id"]] = lean_angle(pts[0], pts[1], W, H)
        else:
            out[d["id"]] = tilt_angle(pts[0], pts[1], W, H)
    return out


# --------------------------------------------------------------------------
# 3D-style mannequin renderer
# Every body part is a shaded capsule / ellipsoid / trunk, painted far-to-near
# using MediaPipe's depth values, then lit with a key light, specular highlight
# and a cool rim light. It is posed from the reference photo's landmarks, so
# elbow, knee, neck, wrist and ankle angles follow the photo.
# --------------------------------------------------------------------------
LIGHT = np.array([-0.45, -0.55, 0.70], np.float32)
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0, 0, 1], np.float32)
HALF /= np.linalg.norm(HALF)
RIM = np.array([175, 215, 255], np.float32)

C_TOP = np.array([84, 150, 245], np.float32)      # jacket: torso, arms
C_BOTTOM = np.array([52, 84, 190], np.float32)    # trousers: pelvis, legs
C_HEAD = np.array([196, 218, 250], np.float32)
C_HAND = np.array([205, 226, 252], np.float32)
C_FOOT = np.array([38, 48, 78], np.float32)


def _shade(nx, ny, nz, base, dk):
    lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    spec = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0, 1) ** 30
    rim = (1.0 - nz) ** 2.5
    k = (0.34 + 0.78 * lam) * dk
    col = base[None, :] * k[:, None]
    col += 62.0 * spec[:, None] + 0.30 * RIM[None, :] * rim[:, None]
    return np.clip(col, 0, 255)


def _grid(pts, pad, W, H):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = int(max(0, np.floor(min(xs) - pad))), int(min(W, np.ceil(max(xs) + pad) + 1))
    y0, y1 = int(max(0, np.floor(min(ys) - pad))), int(min(H, np.ceil(max(ys) + pad) + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    return x0, y0, x1, y1, xx, yy


def _paint(canvas, box, inside, nx, ny, nz, base, dk):
    x0, y0, x1, y1 = box
    region = canvas[y0:y1, x0:x1]
    region[inside, :3] = _shade(nx[inside], ny[inside], nz[inside], base, dk)
    region[inside, 3] = 255.0


def _capsule(canvas, a, b, ra, rb, base, dk=1.0):
    H, W = canvas.shape[:2]
    g = _grid([a, b], max(ra, rb) + 2, W, H)
    if g is None:
        return
    x0, y0, x1, y1, xx, yy = g
    ab = b - a
    L2 = float(ab @ ab)
    t = (np.zeros_like(xx) if L2 < 1e-6 else
         np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2, 0, 1))
    vx = xx - (a[0] + t * ab[0])
    vy = yy - (a[1] + t * ab[1])
    r = np.maximum(ra + t * (rb - ra), 1e-3)
    d2 = vx * vx + vy * vy
    inside = d2 <= r * r
    if inside.any():
        nz = np.sqrt(np.clip(1 - d2 / (r * r), 0, 1))
        _paint(canvas, (x0, y0, x1, y1), inside, vx / r, vy / r, nz, base, dk)


def _ellipsoid(canvas, c, u, ry, rx, base, dk=1.0):
    H, W = canvas.shape[:2]
    g = _grid([c], max(ry, rx) + 2, W, H)
    if g is None:
        return
    x0, y0, x1, y1, xx, yy = g
    v = np.array([-u[1], u[0]], np.float32)
    dx, dy = xx - c[0], yy - c[1]
    ly = (dx * u[0] + dy * u[1]) / ry
    lx = (dx * v[0] + dy * v[1]) / rx
    e = lx * lx + ly * ly
    inside = e <= 1
    if inside.any():
        nz = np.sqrt(np.clip(1 - e, 0, 1))
        _paint(canvas, (x0, y0, x1, y1), inside,
               lx * v[0] + ly * u[0], lx * v[1] + ly * u[1], nz, base, dk)


def _trunk(canvas, top, bot, ts, hws, base, dk=1.0):
    H, W = canvas.shape[:2]
    axis = bot - top
    L = float(np.linalg.norm(axis))
    if L < 1:
        return
    u = axis / L
    v = np.array([-u[1], u[0]], np.float32)
    hm = max(hws)
    corners = [top + u * ts[0] * L + v * hm, top + u * ts[0] * L - v * hm,
               top + u * ts[-1] * L + v * hm, top + u * ts[-1] * L - v * hm]
    g = _grid(corners, 2, W, H)
    if g is None:
        return
    x0, y0, x1, y1, xx, yy = g
    dx, dy = xx - top[0], yy - top[1]
    along = (dx * u[0] + dy * u[1]) / L
    lat = dx * v[0] + dy * v[1]
    hw = np.interp(along, ts, hws, left=0, right=0).astype(np.float32)
    inside = (hw > 0.5) & (np.abs(lat) <= hw)
    if inside.any():
        s = lat / np.maximum(hw, 1e-3)
        nz = np.sqrt(np.clip(1 - s * s, 0, 1))
        _paint(canvas, (x0, y0, x1, y1), inside, s * v[0], s * v[1], nz, base, dk)


def build_mannequin(lm, w, h, flip=True, ss=2):
    """Render the mannequin for landmarks `lm` of a w x h photo.
    Returns dict(rgb=premultiplied float32 (h,w,3), alpha=float32 (h,w), w, h).
    The sprite keeps the photo's exact dimensions, so nothing is cropped."""
    W, H = int(w * ss), int(h * ss)
    canvas = np.zeros((H, W, 4), np.float32)

    def xy(i):
        p = lm[i]
        return np.array([((1 - p.x) if flip else p.x) * W, p.y * H], np.float32)

    def z(i):
        return lm[i].z

    Sm = (xy(11) + xy(12)) / 2
    Hm = (xy(23) + xy(24)) / 2
    S = max(float(np.linalg.norm(Sm - Hm)), 0.05 * min(W, H))   # torso length = unit
    z_trunk = float(np.mean([z(11), z(12), z(23), z(24)]))

    parts = []                                    # (depth, fn)

    def add(depth, fn):
        parts.append((depth, fn))      # fn(dk): dk darkens parts that are farther away

    # ---- head + neck
    face = np.mean([xy(i) for i in range(11)], axis=0)
    u = face - Sm
    nu = float(np.linalg.norm(u))
    u = u / nu if nu > 1e-3 else np.array([0, -1], np.float32)
    head_c = face + u * 0.05 * S
    ry, rx = 0.215 * S, 0.158 * S
    neck_z = z_trunk - 0.01
    add(neck_z, lambda dk: _capsule(canvas, Sm + u * 0.02 * S, head_c - u * 0.7 * ry,
                                    0.080 * S, 0.074 * S, C_HEAD * 0.93, dk))
    add(neck_z - 0.01, lambda dk: _ellipsoid(canvas, head_c, u.astype(np.float32), ry, rx, C_HEAD, dk))

    # ---- trunk (jacket) and pelvis (trousers)
    sw = float(np.linalg.norm(xy(11) - xy(12)))
    hipw = float(np.linalg.norm(xy(23) - xy(24)))
    hw_sh = max(0.5 * sw * 0.82, 0.20 * S)
    hw_ch = max(0.5 * sw * 0.74, 0.21 * S)
    hw_wa = max(0.5 * hipw * 0.9 + 0.05 * S, 0.18 * S)
    hw_hi = max(0.5 * hipw + 0.12 * S, 0.20 * S)
    ts = [-0.13, -0.09, -0.045, 0.0, 0.12, 0.38, 0.65, 0.88]
    hws = [1.0, 0.55 * hw_sh, 0.85 * hw_sh, hw_sh, hw_ch, hw_ch * 0.98, hw_wa, hw_hi * 0.96]
    add(z_trunk + 0.01, lambda dk: _capsule(canvas, xy(23), xy(24), 0.15 * S, 0.15 * S, C_BOTTOM, dk))
    add(z_trunk, lambda dk: _trunk(canvas, Sm, Hm, ts, hws, C_TOP, dk))

    # ---- arms and hands
    for s in (0, 1):
        sh, el, wr = xy(11 + s), xy(13 + s), xy(15 + s)
        add((z(11 + s) + z(13 + s)) / 2,
            lambda dk, sh=sh, el=el: _capsule(canvas, sh, el, 0.115 * S, 0.084 * S, C_TOP, dk))
        add((z(13 + s) + z(15 + s)) / 2,
            lambda dk, el=el, wr=wr: _capsule(canvas, el, wr, 0.084 * S, 0.060 * S, C_TOP, dk))
        fingers = np.mean([xy(17 + s), xy(19 + s), xy(21 + s)], axis=0)
        d = fingers - wr
        dl = float(np.linalg.norm(d))
        if dl < 0.05 * S or lm[19 + s].visibility < 0.15:
            fd = wr - el
            d, dl = fd, float(np.linalg.norm(fd)) + 1e-6
            hand_len = 0.26 * S
        else:
            hand_len = float(np.clip(dl * 1.55, 0.16 * S, 0.34 * S))
        tip = wr + d / dl * hand_len
        add(z(15 + s) - 0.005,
            lambda dk, wr=wr, tip=tip: _capsule(canvas, wr, tip, 0.058 * S, 0.044 * S, C_HAND, dk))

    # ---- legs and feet
    for s in (0, 1):
        hp, kn, an = xy(23 + s), xy(25 + s), xy(27 + s)
        add((z(23 + s) + z(25 + s)) / 2,
            lambda dk, hp=hp, kn=kn: _capsule(canvas, hp, kn, 0.150 * S, 0.102 * S, C_BOTTOM, dk))
        add((z(25 + s) + z(27 + s)) / 2,
            lambda dk, kn=kn, an=an: _capsule(canvas, kn, an, 0.102 * S, 0.066 * S, C_BOTTOM, dk))
        heel, toe = xy(29 + s), xy(31 + s)
        add(z(27 + s) - 0.005,
            lambda dk, an=an, heel=heel: _capsule(canvas, an, heel, 0.066 * S, 0.058 * S, C_FOOT, dk))
        add(z(31 + s) - 0.005,
            lambda dk, heel=heel, toe=toe: _capsule(canvas, heel, toe, 0.058 * S, 0.046 * S, C_FOOT, dk))

    # farthest first; the far side is also slightly darker for a depth cue
    zs = [d for d, _ in parts]
    zmin, zmax = min(zs), max(zs)
    span = max(zmax - zmin, 1e-3)
    for depth, fn in sorted(parts, key=lambda p: -p[0]):
        fn(1.0 - 0.20 * (depth - zmin) / span)

    # crisp light outline around the whole silhouette
    mask = (canvas[..., 3] > 127).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, cnts, -1, (235, 245, 255, 255), int(1.6 * ss), cv2.LINE_AA)

    a = canvas[..., 3] / 255.0
    pre = canvas[..., :3] * a[..., None]
    rgb = cv2.resize(pre, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    alpha = cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    return dict(rgb=rgb, alpha=alpha, w=w, h=h)


def preview_on_dark(sprite):
    """The mannequin on a dark studio background, at the photo's own size."""
    h, w = sprite["h"], sprite["w"]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = (28 + 22 * yy) * np.ones((h, w, 3), np.float32)
    out = bg * (1 - sprite["alpha"][..., None]) + sprite["rgb"]
    return np.clip(out, 0, 255).astype(np.uint8)          # RGB


def img_tag(rgb, ext=".jpg"):
    """<img> with the picture embedded in the page (full picture, original aspect)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, bgr, params)
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    b64 = base64.b64encode(buf.tobytes()).decode()
    return (f'<img src="data:{mime};base64,{b64}" '
            'style="width:100%;height:auto;display:block;border-radius:10px;">')


@st.cache_resource(show_spinner="Analysing the reference photo...")
def load_reference(view):
    """Load the hard-coded reference photo, find its pose, build the mannequin."""
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
        lm, used = detect_static(bgr)
        out = dict(img=bgr, lm=lm, path=str(path), tried=tried, model=used,
                   sprite=None, preview=None, preview_html="", photo_html="")
        if lm is not None:
            h, w = bgr.shape[:2]
            out["sprite"] = build_mannequin(lm, w, h, flip=True)
            out["preview"] = preview_on_dark(out["sprite"])
            out["preview_html"] = img_tag(out["preview"], ".png")
            out["photo_html"] = img_tag(cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB))
        return out
    return dict(img=None, lm=None, path=None, tried=tried, model=None,
                sprite=None, preview=None, preview_html="", photo_html="")


# --------------------------------------------------------------------------
# Drawing helpers for the live video
# --------------------------------------------------------------------------
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
        self.pose, self.complexity = None, None
        self.cfg = None          # set from the Streamlit script on each rerun
        self.view = None
        self.state = {}          # hysteresis memory per angle
        self._ghost_key, self._ghost = None, None

    def _ensure_pose(self, want):
        if self.pose is None or want != self.complexity:
            if self.pose is not None:
                self.pose.close()
            self.pose, _used = make_pose(want)
            self.complexity = want

    def _ghost_for(self, sprite, view, W, H):
        """Fit the whole reference picture inside the frame: uniform scale,
        centred, never cropped."""
        key = (view, W, H)
        if key != self._ghost_key:
            s = min(W / sprite["w"], H / sprite["h"])
            nw = max(1, min(W, int(round(sprite["w"] * s))))
            nh = max(1, min(H, int(round(sprite["h"] * s))))
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            rgb = cv2.resize(sprite["rgb"], (nw, nh), interpolation=interp)
            alpha = cv2.resize(sprite["alpha"], (nw, nh), interpolation=interp)
            ox, oy = (W - nw) // 2, (H - nh) // 2
            self._ghost = (rgb * GHOST_ALPHA, 1.0 - alpha[..., None] * GHOST_ALPHA,
                           (nw, nh, ox, oy))
            self._ghost_key = key
        return self._ghost

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = self.cfg
        if cfg is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        self._ensure_pose(cfg["complexity"])

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

        # ---- reference mannequin (always mirrored, whole picture visible)
        sprite, ref_lm = cfg["sprite"], cfg["ref_lm"]
        if sprite is not None and cfg["show_model"]:
            prgb, keep, (nw, nh, ox, oy) = self._ghost_for(sprite, view, W, H)
            roi = img[oy:oy + nh, ox:ox + nw].astype(np.float32)
            img[oy:oy + nh, ox:ox + nw] = np.clip(roi * keep + prgb, 0, 255).astype(np.uint8)

        if sprite is not None and ref_lm is not None and cfg["show_ref_skel"]:
            _, _, (nw, nh, ox, oy) = self._ghost_for(sprite, view, W, H)

            def rp(spec):
                p = resolve(ref_lm, spec[0], spec[1], cfg["ref_side"], cfg["ref_order"])
                return int((1 - p.x) * nw + ox), int(p.y * nh + oy)

            for a_, b_ in BONES[view]:
                cv2.line(img, rp(a_), rp(b_), BLUE, 2, cv2.LINE_AA)

        banner, banner_color = f"{view} view: no person detected", GREY
        if res.pose_landmarks:
            lm = to_landmarks(res)
            side = pick_side(lm) if cfg["side"] == "auto" else int(cfg["side"])
            order = chain_order(lm, flip=False)

            for a_, b_ in BONES[view]:
                pa = resolve(lm, a_[0], a_[1], side, order)
                pb = resolve(lm, b_[0], b_[1], side, order)
                if vis_ok(pa) and vis_ok(pb):
                    A = (int(pa.x * W), int(pa.y * H))
                    B = (int(pb.x * W), int(pb.y * H))
                    cv2.line(img, A, B, (30, 30, 30), 6, cv2.LINE_AA)
                    cv2.line(img, A, B, WHITE, 3, cv2.LINE_AA)

            cur = compute_angles(lm, W, H, side, order, defs)
            ref_angles = cfg["ref_angles"]
            cx = np.mean([lm[11].x, lm[12].x, lm[23].x, lm[24].x]) * W
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
                        self.state[d["id"]] = "bad" if diff > TOL_DEG + HYST else "ok"
                    else:
                        self.state[d["id"]] = "ok" if diff <= TOL_DEG else "bad"
                    ok = self.state[d["id"]] == "ok"
                    color = GREEN if ok else RED
                    counted += 1
                    ok_n += ok
                    text = f"{d['label']} {c:.0f} (ref {r:.0f})"

                pts = [resolve(lm, n, w, side, order) for n, w in d["pts"]]
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
                if ax < cx:
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
.block-container {padding-top: 3.4rem; max-width: 1400px;}
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
  <p>Match the reference model. Each joint turns green when its angle is within tolerance.</p>
</div>
""", unsafe_allow_html=True)

view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed")
defs = DEFS[view]
ref = load_reference(view)
ref_img, ref_lm, sprite = ref["img"], ref["lm"], ref["sprite"]

with st.sidebar:
    st.subheader("Settings")
    model_label = st.selectbox("Pose model", list(MODEL_CHOICES), index=0,
                               help="Heavy is the most accurate MediaPipe pose model. "
                                    "Choose Full if the live video feels slow.")
    complexity = MODEL_CHOICES[model_label]
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
        show_model = st.checkbox("Show reference model", value=True)
        show_ref_skel = st.checkbox("Show reference skeleton", value=False)
    st.caption(f"In range = within ±{TOL_DEG:.0f}° of the reference. "
               "L / R = left / right half of the screen.")

ref_angles, ref_side, ref_order = None, 0, (0, 1)
if ref_lm is not None:
    ref_side = pick_side(ref_lm) if side == "auto" else int(side)
    ref_order = chain_order(ref_lm, flip=True)      # the reference is always mirrored
    h, w = ref_img.shape[:2]
    ref_angles = compute_angles(ref_lm, w, h, ref_side, ref_order, defs)

left, right = st.columns([3, 1.2], gap="large")

with left:
    st.markdown(
        '<div class="legend">'
        f'<span><i style="background:#1faa59"></i>Within ±{TOL_DEG:.0f}°</span>'
        '<span><i style="background:#e0383e"></i>Out of range</span>'
        '<span><i style="background:#666"></i>Not visible</span></div>',
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
        tab_model, tab_photo = st.tabs(["Reference model", "Source photo"])
        with tab_model:
            st.markdown(ref["preview_html"], unsafe_allow_html=True)
        with tab_photo:
            st.markdown(ref["photo_html"], unsafe_allow_html=True)
        st.caption("Both are shown mirrored, exactly as the overlay appears. "
                   f"Reference analysed with the {'Heavy' if ref['model'] == 2 else 'Full'} model.")
        rows = [{"Angle": d["label"],
                 "Reference": "n/a" if ref_angles[d["id"]] is None
                 else f"{ref_angles[d['id']]:.0f}°"}
                for d in defs if d["id"] in enabled]
        if rows:
            st.table(pd.DataFrame(rows).set_index("Angle"))

if ctx.video_processor:
    ctx.video_processor.cfg = dict(
        view=view, sprite=sprite, ref_lm=ref_lm, ref_angles=ref_angles,
        ref_side=ref_side, ref_order=ref_order, complexity=complexity,
        side=side, enabled=enabled, mirror=mirror,
        show_model=show_model, show_ref_skel=show_ref_skel,
    )