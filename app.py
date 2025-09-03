from __future__ import annotations
import os
import sys
import time
import argparse
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np

############################################################
# ====== KONFIGURASI & KONSTANTA ======
############################################################

# --- Default yang bisa diubah via CLI ---
DEFAULT_VIEW_DWELL_SEC = 2.0       # Detik minimal tatap layar agar dihitung "view"
DEFAULT_WINDOW_SEC = 60            # Interval agregasi snapshot (detik)
DEFAULT_IOU_ASSOC_TH = 0.3         # Ambang IoU untuk asosiasi track wajah
DEFAULT_TRACK_EXP_SEC = 2.0        # Track kadaluarsa jika wajah hilang > X detik
DEFAULT_PROBE_MAX = 6              # Probe kamera dari indeks 0 hingga 5

# --- Konstanta Aplikasi ---
EXPORT_DIR = "exports"
AGE_BINS = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
AGE_UNKNOWN = "unknown"
MIN_FACE_SIZE = (60, 60)

# --- Konstanta Visualisasi ---
COLOR_LOOKING = (0, 200, 0)       # Hijau
COLOR_TRACKED = (80, 80, 80)        # Abu-abu
FONT = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs(EXPORT_DIR, exist_ok=True)

############################################################
# ====== ESTIMATOR USIA (ONNX - Opsional) ======
############################################################
try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

class AgeEstimator:
    def __init__(self, model_path: str, input_size: int = 224):
        if not _HAS_ORT:
            raise RuntimeError("onnxruntime belum terpasang. Jalankan: pip install onnxruntime")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model usia tidak ditemukan: {model_path}")
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]) # type: ignore
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[None, ...]

    def predict_bin(self, face_bgr: np.ndarray) -> str:
        if face_bgr.size == 0:
            return AGE_UNKNOWN
        preprocessed_face = self._preprocess(face_bgr)
        output = self.sess.run(None, {self.input_name: preprocessed_face})[0]
        idx = int(np.argmax(output))
        return AGE_BINS[idx] if 0 <= idx < len(AGE_BINS) else AGE_UNKNOWN

############################################################
# ====== MODEL HAAR (Lazy Loading) ======
############################################################
FACE_CASCADE: Optional[cv2.CascadeClassifier] = None
SMILE_CASCADE: Optional[cv2.CascadeClassifier] = None

def _load_cascade(name: str) -> cv2.CascadeClassifier:
    path = os.path.join(cv2.data.haarcascades, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File Haar Cascade tidak ditemukan di path OpenCV: {name}")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise IOError(f"Gagal memuat Haar Cascade: {name}")
    return cascade

def ensure_cascades_loaded():
    global FACE_CASCADE, SMILE_CASCADE
    if FACE_CASCADE is None:
        FACE_CASCADE = _load_cascade('haarcascade_frontalface_default.xml')
    if SMILE_CASCADE is None:
        SMILE_CASCADE = _load_cascade('haarcascade_smile.xml')

############################################################
# ====== UTILITAS & DATA STRUCTURE ======
############################################################
def iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    return float(inter_area / (boxA_area + boxB_area - inter_area + 1e-6))

@dataclass
class Track:
    box: Tuple[int,int,int,int]
    last_seen: float
    is_looking: bool = False
    look_start: Optional[float] = None
    view_counted: bool = False

    def update(self, box: Tuple[int,int,int,int]):
        self.box = box
        self.last_seen = time.time()

############################################################
# ====== KLASIFIKASI EKSPRESI (HAAR) ======
############################################################
def classify_expression_haar(gray_roi: np.ndarray) -> str:
    if SMILE_CASCADE is None:
        return "unknown"
    smiles = SMILE_CASCADE.detectMultiScale(gray_roi, scaleFactor=1.7, minNeighbors=20)
    return "smile" if len(smiles) > 0 else "neutral"

############################################################
# ====== AGREGATOR METRIK ======
############################################################
def _safe_rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator > 0 else 0.0

class Metrics:
    def __init__(self):
        self.history: deque = deque(maxlen=3600)  # Simpan ringkasan per snapshot hingga 1 jam
        self.today_views = 0
        self.today_impr = 0
        self._lock = threading.Lock()
        self.reset_window()

    def reset_window(self):
        self.window_start = time.time()
        self.impressions = 0
        self.views = 0
        self.dwell_accum = 0.0
        self.dwell_count = 0
        self.age_bins: Dict[str, int] = defaultdict(int)
        self.expr_bins: Dict[str, int] = defaultdict(int)

    def add_impression(self):
        with self._lock:
            self.impressions += 1
            self.today_impr += 1

    def add_view(self):
        with self._lock:
            self.views += 1
            self.today_views += 1

    def add_dwell(self, sec: float):
        with self._lock:
            self.dwell_accum += sec
            self.dwell_count += 1

    def add_demographics(self, age: str, expr: str):
        with self._lock:
            self.age_bins[age] += 1
            self.expr_bins[expr] += 1

    def _current_snapshot_dict(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = now - self.window_start
        rate = _safe_rate(self.views, self.impressions)
        avg_dwell = _safe_rate(self.dwell_accum, self.dwell_count)
        return {
            "ts": int(now),
            "window_sec": int(elapsed),
            "impressions": self.impressions,
            "views": self.views,
            "view_rate": rate,
            "avg_dwell": avg_dwell,
            "age_dist": dict(self.age_bins),
            "expr_dist": dict(self.expr_bins),
        }

    def snapshot_and_reset(self) -> Dict[str, Any]:
        with self._lock:
            snap = self._current_snapshot_dict()
            self.history.append(snap)
            self.reset_window()
            return snap

    def peek_current(self) -> Dict[str, Any]:
        with self._lock:
            data = self._current_snapshot_dict()
            data["today"] = {
                "impressions": self.today_impr,
                "views": self.today_views,
                "view_rate": _safe_rate(self.today_views, self.today_impr)
            }
            return data

    def export_to_csv(self, snap: Dict[str, Any]):
        path = os.path.join(EXPORT_DIR, time.strftime("%Y%m%d") + ".csv")
        header = (
            "ts,window_sec,impressions,views,view_rate,avg_dwell,"
            "age_13_17,age_18_24,age_25_34,age_35_44,age_45_54,age_55p,"
            "expr_neutral,expr_smile,expr_unknown\n"
        )
        row_values = [
            snap["ts"], snap["window_sec"], snap["impressions"], snap["views"],
            f"{snap['view_rate']:.4f}", f"{snap['avg_dwell']:.3f}",
            snap["age_dist"].get("13-17", 0), snap["age_dist"].get("18-24", 0),
            snap["age_dist"].get("25-34", 0), snap["age_dist"].get("35-44", 0),
            snap["age_dist"].get("45-54", 0), snap["age_dist"].get("55+", 0),
            snap["expr_dist"].get("neutral", 0), snap["expr_dist"].get("smile", 0),
            snap["expr_dist"].get("unknown", 0)
        ]
        row = ",".join(map(str, row_values)) + "\n"

        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            f.write(row)
        print(f"[SNAPSHOT] Data tersimpan di {path}")

############################################################
# ====== SERVER WEB (Flask - Opsional) ======
############################################################
try:
    from flask import Flask, jsonify, Response, render_template_string
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

class WebServer:
    def __init__(self, metrics: Metrics, host: str, port: int, refresh_ms: int):
        if not _HAS_FLASK:
            raise RuntimeError("Flask belum terpasang. Jalankan: pip install flask")
        self.metrics = metrics
        self.app = Flask(__name__, template_folder="templates")
        self.host = host
        self.port = port
        self.refresh_ms = refresh_ms
        self._bind_routes()

    def _bind_routes(self):
        @self.app.route("/metrics")
        def _get_metrics():
            return jsonify({
                "current": self.metrics.peek_current(),
                "last_snapshot": self.metrics.history[-1] if self.metrics.history else None,
                "history_len": len(self.metrics.history),
            })

        @self.app.route("/")
        def _render_dashboard():
            try:
                with open(os.path.join(self.app.template_folder, 'index.html'), 'r') as f:
                    template_str = f.read()
                current_data = self.metrics.peek_current()
                html = render_template_string(
                    template_str,
                    current=current_data,
                    today=current_data.get('today', {}),
                    history=list(self.metrics.history)[-30:],
                    refresh_ms=self.refresh_ms
                )
                return Response(html, mimetype='text/html')
            except FileNotFoundError:
                return "Template 'index.html' tidak ditemukan di folder 'templates'", 404

    def start(self):
        thread = threading.Thread(target=lambda: self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False))
        thread.daemon = True
        thread.start()
        print(f"[WEB] Dasbor berjalan di http://{self.host}:{self.port}")

############################################################
# ====== SUMBER FRAME (Kamera, Video, Sintetis) ======
############################################################
class FrameSource:
    def read(self) -> Tuple[bool, np.ndarray]:
        raise NotImplementedError
    def release(self):
        pass
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if FACE_CASCADE is None:
            return []
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE)
        return [(int(x), int(y), int(w), int(h)) for (x,y,w,h) in faces]

class CameraSource(FrameSource):
    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Tidak dapat membuka kamera indeks {index}")
    def read(self) -> Tuple[bool, np.ndarray]:
        return self.cap.read()
    def release(self):
        self.cap.release()

class VideoSource(FrameSource):
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File video tidak ditemukan: {path}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Tidak dapat membuka video: {path}")
    def read(self) -> Tuple[bool, np.ndarray]:
        return self.cap.read()
    def release(self):
        self.cap.release()

class SyntheticSource(FrameSource):
    def __init__(self, w: int = 1280, h: int = 720, n: int = 2):
        self.w, self.h, self.n = w, h, n
        rng = np.random.default_rng(42)
        self.pos = rng.integers(low=[100,100], high=[w-200, h-200], size=(n,2)).astype(np.float32)
        self.vel = rng.uniform(low=-3.0, high=3.0, size=(n,2)).astype(np.float32)
        self.size = np.array([[120,120]] * n, dtype=np.int32)

    def read(self) -> Tuple[bool, np.ndarray]:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.n):
            self.pos[i] += self.vel[i]
            for d in [0,1]:
                if not (10 < self.pos[i, d] < (self.w if d==0 else self.h) - self.size[i, d] - 10):
                    self.vel[i, d] *= -1
        return True, frame

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        faces = []
        for i in range(self.n):
            x, y = int(self.pos[i,0]), int(self.pos[i,1])
            w, h = int(self.size[i,0]), int(self.size[i,1])
            faces.append((x, y, w, h))
        return faces

############################################################
# ====== LOGIKA UTAMA APLIKASI ======
############################################################
def update_tracks(tracks: List[Track], faces: List[Tuple[int,int,int,int]], assoc_iou: float, track_exp: float) -> List[Track]:
    """Memperbarui, menambah, dan menghapus tracks wajah berdasarkan deteksi baru."""
    now = time.time()
    for box in faces:
        best_iou = 0.0
        best_idx = -1
        for i, tr in enumerate(tracks):
            score = iou(box, tr.box)
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_iou > assoc_iou and best_idx != -1:
            tracks[best_idx].update(box)
        else:
            tracks.append(Track(box=box, last_seen=now))
            # Wajah baru adalah impression baru
    return [t for t in tracks if (now - t.last_seen) < track_exp]

def process_frame_metrics(tracks: List[Track], frame: np.ndarray, metrics: Metrics, age_est: Optional[AgeEstimator], dwell_th: float):
    """Menghitung semua metrik (dwell, view, demografi) untuk setiap track aktif."""
    now = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for tr in tracks:
        x, y, w, h = tr.box
        face_roi_gray = gray[y:y+h, x:x+w]

        # Cek Dwell Time & View
        if not tr.is_looking:
            tr.is_looking = True
            tr.look_start = now
        elif tr.look_start and not tr.view_counted and (now - tr.look_start) >= dwell_th:
            metrics.add_view()
            metrics.add_dwell(now - tr.look_start)
            tr.view_counted = True

        # Demografi
        expr = classify_expression_haar(face_roi_gray)
        age = AGE_UNKNOWN
        if age_est and w > 0 and h > 0:
            face_roi_bgr = frame[y:y+h, x:x+w]
            age = age_est.predict_bin(face_roi_bgr)
        metrics.add_demographics(age, expr)

def draw_overlays(frame: np.ndarray, tracks: List[Track]):
    for tr in tracks:
        x, y, w, h = tr.box
        color = COLOR_LOOKING if tr.is_looking else COLOR_TRACKED
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

def run_app(args: argparse.Namespace):
    """Menginisialisasi semua komponen dan menjalankan loop pemrosesan utama."""
    source = setup_source(args)
    metrics = Metrics()
    age_estimator = setup_age_estimator(args)
    tracks: List[Track] = []
    last_snapshot_time = time.time()

    if args.web:
        ws = WebServer(metrics, args.web_host, args.web_port, args.web_refresh_ms)
        ws.start()

    print_startup_message(args)

    try:
        while True:
            ok, frame = source.read()
            if not ok:
                print("[INFO] Sumber video selesai atau terputus.")
                break

            faces = source.detect_faces(frame)
            metrics.add_impression() # Tambah 1 impression per frame dengan wajah

            tracks = update_tracks(tracks, faces, args.assoc_iou, args.track_exp_sec)
            process_frame_metrics(tracks, frame, metrics, age_estimator, args.dwell)

            if not args.no_window:
                draw_overlays(frame, tracks)
                cv2.imshow("Ads Audience Analytics", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if (time.time() - last_snapshot_time) >= args.window_sec:
                snap = metrics.snapshot_and_reset()
                metrics.export_to_csv(snap)
                last_snapshot_time = time.time()

    finally:
        source.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        print("[INFO] Aplikasi berhenti.")

############################################################
# ====== FUNGSI SETUP & MAIN ======
############################################################

############################################################
# ====== UNIT TESTS ======
############################################################
def _almost_equal(a: float, b: float, eps: float=1e-6) -> bool:
    return abs(a-b) < eps

def run_tests():
    """Jalankan semua unit test internal."""
    print("[TEST] Menjalankan unit tests...")
    # Test IOU
    assert _almost_equal(iou((0,0,10,10), (0,0,10,10)), 1.0)
    assert _almost_equal(round(iou((0,0,10,10), (5,5,10,10)), 3), 0.143) # Adjusted for correct math
    assert _almost_equal(iou((0,0,10,10), (20,20,5,5)), 0.0)
    print("[PASS] test_iou_basic")

    # Test Metrics
    m = Metrics()
    m.add_impression()
    m.add_impression()
    m.add_view()
    m.add_dwell(3.0)
    m.add_demographics('18-24', 'smile')
    snap = m.snapshot_and_reset()
    assert snap['impressions'] == 2
    assert snap['views'] == 1
    assert _almost_equal(snap['avg_dwell'], 3.0)
    assert snap['age_dist'].get('18-24', 0) == 1
    assert snap['expr_dist'].get('smile', 0) == 1
    assert m.impressions == 0 # Check reset
    print("[PASS] test_metrics_snapshot")

    # Test CSV
    m.export_to_csv(snap)
    path = os.path.join(EXPORT_DIR, time.strftime("%Y%m%d") + ".csv")
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert 'impressions' in content and '2' in content
    print("[PASS] test_csv_export")

    # Test Peek
    m.add_impression()
    cur = m.peek_current()
    assert set(['ts','window_sec','impressions','views','view_rate','avg_dwell','age_dist','expr_dist','today']).issubset(cur.keys())
    assert cur['impressions'] == 1
    print("[PASS] test_peek_current")

    print("\nSemua test berhasil dilewati.")

def list_available_cameras(max_to_probe: int) -> List[int]:
    available = []
    for i in range(max_to_probe):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            if cap.read()[0]:
                available.append(i)
            cap.release()
    return available

def setup_source(args: argparse.Namespace) -> FrameSource:
    """Memilih dan menginisialisasi sumber frame berdasarkan argumen CLI."""
    ensure_cascades_loaded()
    if args.video:
        return VideoSource(args.video)
    if args.synthetic:
        return SyntheticSource()
    try:
        return CameraSource(args.camera_index)
    except RuntimeError as e:
        print(f"[WARN] Gagal membuka kamera indeks {args.camera_index}: {e}")
        print(f"[INFO] Mencari kamera lain yang tersedia (0..{DEFAULT_PROBE_MAX-1})...")
        available_cams = list_available_cameras(DEFAULT_PROBE_MAX)
        if available_cams:
            print(f"[INFO] Kamera ditemukan di indeks {available_cams[0]}, menggunakan kamera tersebut.")
            return CameraSource(available_cams[0])
        else:
            print("[WARN] Tidak ada kamera ditemukan. Beralih ke mode sintetis.")
            return SyntheticSource()

def setup_age_estimator(args: argparse.Namespace) -> Optional[AgeEstimator]:
    if not args.age_model:
        return None
    try:
        estimator = AgeEstimator(args.age_model)
        print(f"[INFO] Model estimasi usia berhasil dimuat dari: {args.age_model}")
        return estimator
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[WARN] Gagal memuat model usia: {e}. Fitur usia dinonaktifkan.")
        return None

def print_startup_message(args: argparse.Namespace):
    if args.no_window:
        print("[INFO] Berjalan dalam mode headless. Tidak ada jendela video akan ditampilkan.")
    else:
        print("[INFO] Aplikasi berjalan. Tekan 'q' di jendela video untuk keluar.")

def main():
    parser = argparse.ArgumentParser(description="Ads Audience Analytics - Purwarupa")
    parser.add_argument("--camera-index", type=int, default=0, help="Indeks kamera.")
    parser.add_argument("--video", type=str, help="Path ke file video.")
    parser.add_argument("--synthetic", action="store_true", help="Gunakan sumber data sintetis.")
    parser.add_argument("--list-cams", action="store_true", help="Tampilkan daftar kamera yang tersedia lalu keluar.")
    parser.add_argument("--no-window", action="store_true", help="Jalankan tanpa jendela video (headless).")
    parser.add_argument("--window-sec", type=int, default=DEFAULT_WINDOW_SEC, help=f"Jendela agregasi (detik, default: {DEFAULT_WINDOW_SEC})")
    parser.add_argument("--dwell", type=float, default=DEFAULT_VIEW_DWELL_SEC, help=f"Ambang dwell (detik) untuk view (default: {DEFAULT_VIEW_DWELL_SEC})")
    parser.add_argument("--assoc-iou", type=float, default=DEFAULT_IOU_ASSOC_TH, help=f"Ambang IoU untuk asosiasi track (default: {DEFAULT_IOU_ASSOC_TH})")
    parser.add_argument("--track-exp-sec", type=float, default=DEFAULT_TRACK_EXP_SEC, help=f"Detik kedaluwarsa track (default: {DEFAULT_TRACK_EXP_SEC})")
    parser.add_argument("--web", action="store_true", help="Aktifkan dasbor web (membutuhkan Flask).")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Host untuk dasbor web.")
    parser.add_argument("--web-port", type=int, default=8000, help="Port untuk dasbor web.")
    parser.add_argument("--web-refresh-ms", type=int, default=2000, help="Interval refresh dasbor web (ms).")
    parser.add_argument("--age-model", type=str, help="Path ke model usia ONNX (opsional).")
    parser.add_argument("--run-tests", action="store_true", help="Jalankan unit tests lalu keluar.")
    args = parser.parse_args()

    if args.run_tests:
        run_tests()
        sys.exit(0)

    if args.list_cams:
        cams = list_available_cameras(DEFAULT_PROBE_MAX)
        print("Kamera yang tersedia:", ", ".join(map(str, cams)) if cams else "Tidak ada")
        sys.exit(0)

    run_app(args)

if __name__ == "__main__":
    main()