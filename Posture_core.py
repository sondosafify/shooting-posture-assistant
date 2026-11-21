"""
posture_core.py  -  vision / maths engine for the Air Rifle Posture Assistant.

No Streamlit code in here, so it can be unit-tested and reused.

Pipeline per frame
  1. MediaPipe Pose finds 33 landmarks on the live frame.
  2. The reference photo (letter-boxed, never cropped) is warped so that its
     hip-centre and body length match the shooter -> overlay follows the user,
     only the *posture* difference remains visible.
  3. 5 angles are measured on the live body and on the reference body.
  4. Each live angle is green if |live - reference| <= tolerance, else red.
"""
from __future__ import annotations

import math
import threading
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Colours (BGR for OpenCV)
# --------------------------------------------------------------------------- #
GREEN = (94, 197, 34)
RED = (68, 68, 239)
GREY = (175, 163, 156)
AMBER = (36, 191, 251)
BLUE = (246, 160, 80)
WHITE = (255, 255, 255)
INK = (38, 28, 18)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------------------------------------------------------- #
# Landmarks
# --------------------------------------------------------------------------- #
# name -> (left index, right index)  (MediaPipe naming, anatomical sides)
IDX = {
    "ear": (7, 8), "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28),
}
LEFT_SET = [11, 13, 15, 23, 25, 27]
RIGHT_SET = [12, 14, 16, 24, 26, 28]
SIDE_NAME = {0: "LEFT", 1: "RIGHT"}

_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
          (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
SWAP_ORDER = list(range(33))
for _a, _b in _PAIRS:
    SWAP_ORDER[_a], SWAP_ORDER[_b] = _b, _a

BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]

VIS_MIN = 0.35

# --------------------------------------------------------------------------- #
# Optional coach overrides.  If an entry exists, it REPLACES the reference-photo
# target for that angle:  (view, angle_key): (min_deg, max_deg)
# Example:  ("Side", "knee"): (168, 180),
# --------------------------------------------------------------------------- #
COACH_RANGES: dict[tuple[str, str], tuple[float, float]] = {}


# --------------------------------------------------------------------------- #
# Angle definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    kind: str          # joint | lean | tilt | midlean | spread
    pts: tuple


def specs_for(view: str, side: int) -> list[Spec]:
    """Five angles per view.  Side view uses the limbs nearest the camera."""
    if view == "Side":
        sh, el, wr, hp, kn, an = (IDX[n][side] for n in
                                  ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"))
        return [
            Spec("elbow", "Elbow", "joint", (sh, el, wr)),
            Spec("arm", "Arm lift", "joint", (hp, sh, el)),
            Spec("lean", "Torso lean", "lean", (hp, sh)),
            Spec("hip", "Hip", "joint", (sh, hp, kn)),
            Spec("knee", "Knee", "joint", (hp, kn, an)),
        ]
    # Front / Back: frontal-plane angles (elbow flexion is not reliable here)
    return [
        Spec("shoulders", "Shoulder tilt", "tilt", (11, 12)),
        Spec("hips", "Hip tilt", "tilt", (23, 24)),
        Spec("spine", "Spine lean", "midlean", (23, 24, 11, 12)),
        Spec("head", "Head tilt", "tilt", (7, 8)),
        Spec("stance", "Stance width", "spread", (27, 23, 24, 28)),
    ]


@dataclass
class Measure:
    value: float
    anchor: np.ndarray
    draw: tuple          # ("arc", centre, a, b) | ("line", a, b)


def _deg(rad: float) -> float:
    return math.degrees(rad)


def angle3(a, b, c):
    v1, v2 = np.asarray(a) - b, np.asarray(c) - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return None
    return _deg(math.acos(float(np.clip(np.dot(v1, v2) / n, -1, 1))))


def _tilt(a, b):
    d = np.asarray(b) - a
    ang = _deg(math.atan2(d[1], d[0]))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return ang


def measure(spec: Spec, px: np.ndarray, vis: np.ndarray) -> Measure | None:
    """px: (33,2) pixel coords, vis: (33,) visibility.  None if not reliable."""
    need = list(spec.pts) + ([0] if spec.kind == "lean" else [])
    if any(vis[i] < VIS_MIN for i in need):
        return None
    p = lambda i: px[i]
    k = spec.kind
    if k == "joint":
        a, b, c = (p(i) for i in spec.pts)
        v = angle3(a, b, c)
        return None if v is None else Measure(v, b, ("arc", b, a, c))
    if k == "lean":                       # + = leaning back, away from the target
        hip, sh = p(spec.pts[0]), p(spec.pts[1])
        d = sh - hip
        raw = _deg(math.atan2(d[0], -d[1]))
        face = 1.0 if (px[0][0] - sh[0]) >= 0 else -1.0
        return Measure(-face * raw, sh, ("line", hip, sh))
    if k == "tilt":
        a, b = p(spec.pts[0]), p(spec.pts[1])
        return Measure(_tilt(a, b), (a + b) / 2, ("line", a, b))
    if k == "midlean":
        hm = (p(spec.pts[0]) + p(spec.pts[1])) / 2
        sm = (p(spec.pts[2]) + p(spec.pts[3])) / 2
        d = sm - hm
        return Measure(_deg(math.atan2(d[0], -d[1])), sm, ("line", hm, sm))
    if k == "spread":
        al, ar = p(spec.pts[0]), p(spec.pts[3])
        hm = (p(spec.pts[1]) + p(spec.pts[2])) / 2
        v = angle3(al, hm, ar)
        return None if v is None else Measure(v, hm, ("arc", hm, al, ar))
    return None


def fmt(kind: str, v: float) -> str:
    return f"{v:.0f}" if kind in ("joint", "spread") else f"{v:+.0f}"


def near_side(kp: np.ndarray) -> int:
    """0 = LEFT side of the person faces the camera, 1 = RIGHT."""
    return 0 if kp[LEFT_SET, 3].mean() >= kp[RIGHT_SET, 3].mean() else 1


# --------------------------------------------------------------------------- #
# Pose detection backends
# --------------------------------------------------------------------------- #
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_{name}/float16/latest/pose_landmarker_{name}.task")
_MODEL_DIR = Path.home() / ".cache" / "posture_app"


def _task_model(name: str) -> str:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / f"pose_landmarker_{name}.task"
    if not path.exists():
        urllib.request.urlretrieve(_MODEL_URL.format(name=name), path)
    return str(path)


class Detector:
    """detect(rgb) -> (kp (33,4) normalised x,y,z,vis  |  None,  mask float HxW | None)"""

    def __init__(self, static: bool, complexity: int, segmentation: bool = False):
        import mediapipe as mp
        self.mp, self.static, self.segmentation = mp, static, segmentation
        self._ts = 0
        if hasattr(mp, "solutions"):                      # legacy API (mediapipe<=0.10.2x)
            self.legacy = True
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=static, model_complexity=complexity,
                smooth_landmarks=True, enable_segmentation=segmentation,
                min_detection_confidence=0.4, min_tracking_confidence=0.5)
        else:                                             # Tasks API fallback
            self.legacy = False
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
            name = {0: "lite", 1: "full", 2: "heavy"}[complexity]
            opts = vision.PoseLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=_task_model(name)),
                running_mode=vision.RunningMode.IMAGE if static else vision.RunningMode.VIDEO,
                num_poses=1, output_segmentation_masks=segmentation,
                min_pose_detection_confidence=0.4, min_tracking_confidence=0.5)
            self.pose = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray):
        mp = self.mp
        if self.legacy:
            res = self.pose.process(rgb)
            if not res.pose_landmarks:
                return None, None
            kp = np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark],
                          np.float32)
            mask = res.segmentation_mask if self.segmentation else None
            return kp, mask
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.static:
            res = self.pose.detect(img)
        else:
            self._ts = max(self._ts + 1, int(time.time() * 1000))
            res = self.pose.detect_for_video(img, self._ts)
        if not res.pose_landmarks:
            return None, None
        kp = np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks[0]], np.float32)
        mask = None
        if self.segmentation and res.segmentation_masks:
            mask = res.segmentation_masks[0].numpy_view().astype(np.float32)
        return kp, mask

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Reference photos
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    view: str
    bgr: np.ndarray                  # (h,w,3)
    alpha_cut: np.ndarray            # (h,w) uint8 - shooter cut-out
    kp: np.ndarray                   # (33,4) normalised
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def size(self):                  # (h, w)
        return self.bgr.shape[:2]

    def variant(self, flip: bool, cutout: bool):
        """RGBA image + keypoints (+ near side).  flip = mirror (left-handed shooter)."""
        key = (flip, cutout)
        if key not in self._cache:
            bgr, alpha, kp = self.bgr, self.alpha_cut, self.kp
            if not cutout:
                alpha = np.full_like(alpha, 255)
            if flip:
                bgr, alpha = bgr[:, ::-1], alpha[:, ::-1]
                kp = kp[SWAP_ORDER].copy()
                kp[:, 0] = 1.0 - kp[:, 0]
            rgba = np.dstack([bgr, alpha]).astype(np.uint8)
            self._cache[key] = (np.ascontiguousarray(rgba), kp, near_side(kp))
        return self._cache[key]


def load_reference(view: str, candidates: list[Path]) -> Reference:
    """Read the photo, detect its pose once, build a shooter cut-out."""
    bgr = None
    for p in candidates:
        if Path(p).exists():
            bgr = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                break
    if bgr is None:
        raise FileNotFoundError(f"{view}: image not found. Tried: " + " | ".join(map(str, candidates)))
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    kp = mask = None
    for cx in (2, 1):
        try:
            det = Detector(static=True, complexity=cx, segmentation=True)
            kp, mask = det.detect(rgb)
            det.close()
        except Exception:
            kp = None
        if kp is not None:
            break
    if kp is None:
        raise RuntimeError(f"{view}: no person detected in the reference photo.")
    if mask is None:
        alpha = np.full(bgr.shape[:2], 255, np.uint8)
    else:
        a = np.clip((mask - 0.35) / 0.3, 0, 1)
        alpha = (cv2.GaussianBlur(a, (0, 0), 1.6) * 255).astype(np.uint8)
    return Reference(view, bgr, alpha, kp)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _i(p):
    return (int(round(p[0])), int(round(p[1])))


def draw_skeleton(img, px, vis, color, fs, dots=True):
    for a, b in BONES:
        if vis[a] > VIS_MIN and vis[b] > VIS_MIN:
            cv2.line(img, _i(px[a]), _i(px[b]), color, max(2, int(3 * fs)), cv2.LINE_AA)
    if dots:
        for i in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
            if vis[i] > VIS_MIN:
                cv2.circle(img, _i(px[i]), max(3, int(4 * fs)), color, -1, cv2.LINE_AA)


def draw_measure(img, m: Measure, color, fs):
    th = max(3, int(4 * fs))
    if m.draw[0] == "line":
        cv2.line(img, _i(m.draw[1]), _i(m.draw[2]), color, th, cv2.LINE_AA)
        return
    _, c, a, b = m.draw
    c, a, b = np.asarray(c), np.asarray(a), np.asarray(b)
    for q in (a, b):
        v = q - c
        n = np.linalg.norm(v)
        if n > 1:
            cv2.line(img, _i(c), _i(c + v / n * min(n, 120 * fs)), color, th, cv2.LINE_AA)
    r = int(max(18 * fs, 0.28 * min(np.linalg.norm(a - c), np.linalg.norm(b - c))))
    a1 = _deg(math.atan2(a[1] - c[1], a[0] - c[0]))
    a2 = _deg(math.atan2(b[1] - c[1], b[0] - c[0]))
    d = (a2 - a1 + 180) % 360 - 180
    s, e = (a1, a1 + d) if d >= 0 else (a1 + d, a1)
    cv2.ellipse(img, _i(c), (r, r), 0, s, e, color, th, cv2.LINE_AA)


def draw_tag(img, anchor, away, label, value, target, color, fs, placed, ring=True):
    """Rounded-look label box next to a joint.  `placed` avoids overlaps."""
    H, W = img.shape[:2]
    pad = int(8 * fs)
    s_sm, s_bg = 0.44 * fs, 0.85 * fs
    t_sm, t_bg = max(1, int(fs)), max(2, int(2 * fs))
    (w1, h1), _ = cv2.getTextSize(label, FONT, s_sm, t_sm)
    (w2, h2), _ = cv2.getTextSize(value, FONT, s_bg, t_bg)
    rw = int(7 * fs) if ring else 0
    (w3, h3), _ = cv2.getTextSize(target, FONT, s_sm, t_sm) if target else ((0, 0), 0)
    line2 = w2 + rw + ((int(10 * fs) + w3) if target else 0)
    bw = max(w1, line2) + 2 * pad + int(4 * fs)
    bh = pad + h1 + int(7 * fs) + h2 + pad
    n = np.linalg.norm(away)
    away = away / n if n > 1e-3 else np.array([1.0, 0.0])
    c = np.asarray(anchor) + away * (62 * fs + bw / 2 * abs(away[0]))
    x0 = int(np.clip(c[0] - bw / 2, 4, W - bw - 4))
    y0 = int(np.clip(c[1] - bh / 2, 4, H - bh - 4))
    for _ in range(8):                                   # push down while overlapping
        if any(not (x0 + bw < px or px + pw < x0 or y0 + bh < py or py + ph < y0)
               for px, py, pw, ph in placed):
            y0 = min(H - bh - 4, y0 + bh + 6)
        else:
            break
    placed.append((x0, y0, bw, bh))
    cv2.line(img, _i(anchor), (x0 + bw // 2, y0 + bh // 2), color, max(1, int(fs)), cv2.LINE_AA)
    roi = img[y0:y0 + bh, x0:x0 + bw]
    cv2.addWeighted(roi, 0.30, np.full_like(roi, INK), 0.70, 0, roi)
    cv2.rectangle(img, (x0, y0), (x0 + int(4 * fs), y0 + bh), color, -1)
    tx = x0 + pad + int(4 * fs)
    cv2.putText(img, label, (tx, y0 + pad + h1), FONT, s_sm, (220, 215, 210), t_sm, cv2.LINE_AA)
    by = y0 + pad + h1 + int(7 * fs) + h2
    cv2.putText(img, value, (tx, by), FONT, s_bg, WHITE if value != "--" else GREY, t_bg, cv2.LINE_AA)
    if ring and value != "--":
        cv2.circle(img, (tx + w2 + int(4 * fs), by - h2 + int(2 * fs)), max(2, int(3 * fs)),
                   WHITE, max(1, int(fs)), cv2.LINE_AA)
    if target:
        cv2.putText(img, target, (tx + w2 + rw + int(10 * fs), by), FONT, s_sm, color, t_sm, cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha=0.65):
    roi = img[y:y + h, x:x + w]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, INK), alpha, 0, roi)


def draw_hud(img, view, ok, total, fs):
    x, y, w, h = int(14 * fs), int(14 * fs), int(250 * fs), int(74 * fs)
    _panel(img, x, y, w, h)
    cv2.putText(img, f"{view.upper()} VIEW", (x + 12, y + int(24 * fs)), FONT, 0.55 * fs, (220, 215, 210),
                max(1, int(fs)), cv2.LINE_AA)
    col = GREEN if total and ok == total else (AMBER if ok else RED)
    cv2.putText(img, f"{ok}/{total} in range", (x + 12, y + int(50 * fs)), FONT, 0.7 * fs, WHITE,
                max(2, int(2 * fs)), cv2.LINE_AA)
    bx, by, bw = x + 12, y + int(60 * fs), w - 24
    cv2.rectangle(img, (bx, by), (bx + bw, by + int(6 * fs)), (90, 80, 70), -1)
    if total:
        cv2.rectangle(img, (bx, by), (bx + int(bw * ok / total), by + int(6 * fs)), col, -1)


def draw_banner(img, text, color, fs):
    H, W = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8 * fs, max(2, int(2 * fs)))
    bw, bh = tw + int(40 * fs), th + int(26 * fs)
    x, y = (W - bw) // 2, H - bh - int(22 * fs)
    _panel(img, x, y, bw, bh, 0.78)
    cv2.rectangle(img, (x, y), (x + bw, y + int(4 * fs)), color, -1)
    cv2.putText(img, text, (x + int(20 * fs), y + bh - int(13 * fs)), FONT, 0.8 * fs, WHITE,
                max(2, int(2 * fs)), cv2.LINE_AA)


def composite(frame, overlay_rgba, opacity_pct):
    """Alpha-blend an RGBA overlay on a BGR frame (integer maths, fast)."""
    a = (overlay_rgba[..., 3].astype(np.uint16) * opacity_pct // 100)[..., None]
    out = (frame.astype(np.uint16) * (255 - a) + overlay_rgba[..., :3].astype(np.uint16) * a) // 255
    return out.astype(np.uint8)


def letterbox_rgba(rgba, W, H):
    """Fit the whole image inside W x H without cropping (transparent padding)."""
    h, w = rgba.shape[:2]
    s = min(W / w, H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    res = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 4), np.uint8)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = res
    return canvas, (s, ox, oy)


def body_metrics(px, vis):
    """(hip_centre, torso_len, full_len|None) or None."""
    if any(vis[i] < VIS_MIN for i in (11, 12, 23, 24)):
        return None
    sh, hp = (px[11] + px[12]) / 2, (px[23] + px[24]) / 2
    full = None
    if vis[27] >= VIS_MIN and vis[28] >= VIS_MIN:
        full = float(np.linalg.norm(sh - (px[27] + px[28]) / 2))
    return hp, float(np.linalg.norm(sh - hp)), full


# --------------------------------------------------------------------------- #
# Config + engine
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    view: str = "Side"
    left_handed: bool = False
    mirror_camera: bool = False
    mirror_ref: bool = False
    opacity: int = 45
    tolerance: float = 10.0
    cutout: bool = True
    autofit: bool = True
    show_ref_skeleton: bool = False
    show_live_skeleton: bool = True
    complexity: int = 1


class PostureEngine:
    def __init__(self, refs: dict[str, Reference]):
        self.refs = refs
        self._cfg = Config()
        self._lock = threading.Lock()
        self._det: Detector | None = None
        self._det_cx = None
        self._base_key = None
        self._base = None
        self._fit = None
        self._side = None
        self._ema: dict = {}
        self._stable = 0
        self._fps = 0.0
        self._view_seen = None
        self._result = {"rows": [], "ok": 0, "total": 0, "detected": False, "hint": "", "fps": 0.0}

    # -- thread-safe config / result ------------------------------------- #
    def set_config(self, cfg: Config):
        with self._lock:
            self._cfg = replace(cfg)

    def get_result(self):
        with self._lock:
            return dict(self._result)

    # -- detection --------------------------------------------------------- #
    def _detect(self, img, cx):
        if self._det is None or self._det_cx != cx:
            if self._det:
                self._det.close()
            self._det, self._det_cx = Detector(static=False, complexity=cx), cx
        H, W = img.shape[:2]
        s = min(1.0, 480 / W)
        small = cv2.resize(img, (int(W * s), int(H * s))) if s < 1 else img
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        kp, _ = self._det.detect(rgb)
        if kp is None:
            return None, None
        return kp[:, :2] * np.array([W, H], np.float32), kp[:, 3]

    def _update_side(self, vis):
        l, r = vis[LEFT_SET].mean(), vis[RIGHT_SET].mean()
        if self._side is None:
            self._side = 0 if l >= r else 1
        elif self._side == 0 and r > l + 0.12:
            self._side = 1
        elif self._side == 1 and l > r + 0.12:
            self._side = 0

    def _update_fit(self, cfg, rpx, rvis, lpx, lvis):
        if not cfg.autofit:
            self._fit = np.array([1.0, 0.0, 0.0])
            return self._fit
        if self._fit is None:
            self._fit = np.array([1.0, 0.0, 0.0])
        if lpx is None:
            return self._fit
        rm, lm = body_metrics(rpx, rvis), body_metrics(lpx, lvis)
        if rm is None or lm is None:
            return self._fit
        use_full = rm[2] is not None and lm[2] is not None
        rl, ll = (rm[2], lm[2]) if use_full else (rm[1], lm[1])
        if rl < 20 or ll < 20:
            return self._fit
        k = float(np.clip(ll / rl, 0.4, 2.5))
        t = lm[0] - k * rm[0]
        target = np.array([k, t[0], t[1]])
        self._fit = self._fit * 0.75 + target * 0.25
        return self._fit

    # -- main --------------------------------------------------------------- #
    def process(self, img: np.ndarray) -> np.ndarray:
        t0 = time.time()
        with self._lock:
            cfg = replace(self._cfg)
        if cfg.mirror_camera:
            img = cv2.flip(img, 1)
        H, W = img.shape[:2]
        fs = max(0.6, H / 720)
        if cfg.view != self._view_seen:
            self._ema.clear()
            self._stable = 0
            self._view_seen = cfg.view

        lpx, lvis = self._detect(img, cfg.complexity)
        if lpx is not None and cfg.view == "Side":
            self._update_side(lvis)
        side = self._side if self._side is not None else 1

        ref = self.refs.get(cfg.view)
        rows, hint = [], ""
        rpx_t = rvis = None
        rside = 1
        if ref is not None:
            flip = cfg.left_handed != cfg.mirror_ref
            rgba, rkp, rside = ref.variant(flip, cfg.cutout)
            rh, rw = ref.size
            key = (cfg.view, flip, cfg.cutout, W, H)
            if key != self._base_key:
                self._base, self._geo = letterbox_rgba(rgba, W, H)
                self._base_key = key
            s, ox, oy = self._geo
            rpx = rkp[:, :2] * np.array([rw * s, rh * s]) + np.array([ox, oy])
            rvis = rkp[:, 3]
            k, tx, ty = self._update_fit(cfg, rpx, rvis, lpx, lvis)
            if abs(k - 1) > 1e-3 or abs(tx) > 0.5 or abs(ty) > 0.5:
                M = np.array([[k, 0, tx], [0, k, ty]], np.float32)
                ov = cv2.warpAffine(self._base, M, (W, H), flags=cv2.INTER_LINEAR)
            else:
                ov = self._base
            rpx_t = rpx * k + np.array([tx, ty])
            img = composite(img, ov, cfg.opacity)
        else:
            hint = "Reference image missing"

        # ---- angles ---- #
        mismatch = cfg.view == "Side" and lpx is not None and side != rside
        if mismatch:
            hint = f"Turn so your {SIDE_NAME[rside]} side faces the camera"
        rspecs = {sp.key: sp for sp in specs_for(cfg.view, rside)}
        placed = []
        ok_n = 0
        centre = None
        if lpx is not None:
            vv = [i for i in (11, 12, 23, 24) if lvis[i] > VIS_MIN]
            centre = lpx[vv].mean(axis=0) if vv else np.array([W / 2, H / 2])
            if cfg.show_live_skeleton:
                draw_skeleton(img, lpx, lvis, (255, 255, 255), fs)
        if cfg.show_ref_skeleton and rpx_t is not None:
            draw_skeleton(img, rpx_t, rvis, BLUE, fs, dots=False)

        for sp in specs_for(cfg.view, side):
            tgt = None
            rm = measure(rspecs[sp.key], rpx, rvis) if rpx_t is not None else None
            coach = COACH_RANGES.get((cfg.view, sp.key))
            lm = measure(sp, lpx, lvis) if (lpx is not None and not mismatch) else None
            val = None
            if lm is not None:
                ek = (cfg.view, sp.key)
                val = lm.value if ek not in self._ema else self._ema[ek] * 0.6 + lm.value * 0.4
                self._ema[ek] = val
            if coach:
                tgt_txt = f"{coach[0]:.0f}-{coach[1]:.0f}"
                status = "na" if val is None else ("ok" if coach[0] <= val <= coach[1] else "bad")
            elif rm is not None:
                tgt = rm.value
                tgt_txt = fmt(sp.kind, tgt)
                status = "na" if val is None else ("ok" if abs(val - tgt) <= cfg.tolerance else "bad")
            else:
                tgt_txt, status = "", "na"
            ok_n += status == "ok"
            rows.append({"key": sp.key, "label": sp.label, "kind": sp.kind,
                         "live": val, "target": tgt, "target_txt": tgt_txt, "status": status})
            if lm is not None and not mismatch:
                col = {"ok": GREEN, "bad": RED, "na": GREY}[status]
                draw_measure(img, lm, col, fs)
                draw_tag(img, lm.anchor, lm.anchor - centre, sp.label,
                         fmt(sp.kind, val), f"ref {tgt_txt}" if tgt_txt else "", col, fs, placed)

        total = len(rows)
        draw_hud(img, cfg.view, ok_n, total, fs)
        all_ok = total > 0 and ok_n == total and lpx is not None
        self._stable = self._stable + 1 if all_ok else 0
        if lpx is None:
            draw_banner(img, "Step into the frame - full body visible", AMBER, fs)
            hint = hint or "No shooter detected"
        elif hint:
            draw_banner(img, hint, AMBER, fs)
        elif self._stable >= 6:
            draw_banner(img, "STANCE MATCHED", GREEN, fs)

        dt = max(1e-3, time.time() - t0)
        self._fps = 0.9 * self._fps + 0.1 / dt if self._fps else 1 / dt
        with self._lock:
            self._result = {"rows": rows, "ok": ok_n, "total": total, "detected": lpx is not None,
                            "hint": hint, "fps": self._fps, "side": SIDE_NAME[side]}
        return img


# --------------------------------------------------------------------------- #
# Static reference card (for the UI)  - fixed-size canvas, never crops
# --------------------------------------------------------------------------- #
def render_reference_card(ref: Reference, flip: bool, cutout: bool = False,
                          W: int = 600, H: int = 800) -> tuple[np.ndarray, list[dict]]:
    rgba, kp, side = ref.variant(flip, cutout)
    bg = np.full((H, W, 3), (42, 30, 20), np.uint8)            # BGR of deep navy
    base, (s, ox, oy) = letterbox_rgba(rgba, W, H)
    img = composite(bg, base, 100)
    rh, rw = ref.size
    px = kp[:, :2] * np.array([rw * s, rh * s]) + np.array([ox, oy])
    vis = kp[:, 3]
    fs = H / 720
    draw_skeleton(img, px, vis, BLUE, fs)
    centre = px[[11, 12, 23, 24]].mean(axis=0)
    placed, info = [], []
    for sp in specs_for(ref.view, side):
        m = measure(sp, px, vis)
        if m is None:
            continue
        coach = COACH_RANGES.get((ref.view, sp.key))
        txt = f"{coach[0]:.0f}-{coach[1]:.0f}" if coach else ""
        draw_measure(img, m, BLUE, fs)
        draw_tag(img, m.anchor, m.anchor - centre, sp.label, fmt(sp.kind, m.value), txt, BLUE, fs, placed)
        info.append({"label": sp.label, "value": fmt(sp.kind, m.value)})
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), info