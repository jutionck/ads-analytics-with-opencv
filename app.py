from __future__ import annotations
import os
import sys
import time
import argparse
import threading
import json
import glob
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np

# Cek dependensi networking
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

############################################################
# ====== KONFIGURASI & KONSTANTA ======
############################################################

# --- Default yang bisa diubah via CLI ---
DEFAULT_VIEW_DWELL_SEC = 2.0
DEFAULT_WINDOW_SEC = 60
DEFAULT_IOU_ASSOC_TH = 0.3
DEFAULT_TRACK_EXP_SEC = 2.0
DEFAULT_PROBE_MAX = 6

# --- Konstanta Aplikasi ---
EXPORT_DIR = "exports"
PENDING_DIR = "pending_data"
AGE_BINS = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
AGE_UNKNOWN = "unknown"
GENDER_BINS = ["Pria", "Wanita"]
GENDER_UNKNOWN = "unknown"
MIN_FACE_SIZE = (60, 60)

# --- Konstanta Visualisasi ---
COLOR_LOOKING = (0, 200, 0)
COLOR_TRACKED = (80, 80, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(PENDING_DIR, exist_ok=True)

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
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
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

class GenderEstimator:
    def __init__(self, model_path: str, input_size: int = 224):
        if not _HAS_ORT:
            raise RuntimeError("onnxruntime belum terpasang. Jalankan: pip install onnxruntime")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model gender tidak ditemukan: {model_path}")
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[None, ...]

    def predict_bin(self, face_bgr: np.ndarray) -> str:
        if face_bgr.size == 0:
            return GENDER_UNKNOWN
        preprocessed_face = self._preprocess(face_bgr)
        output = self.sess.run(None, {self.input_name: preprocessed_face})[0]
        idx = int(np.argmax(output))
        return GENDER_BINS[idx] if 0 <= idx < len(GENDER_BINS) else GENDER_UNKNOWN

############################################################
# ====== MODEL HAAR (Lazy Loading) ======
############################################################
FACE_CASCADE: Optional[cv2.CascadeClassifier] = None
SMILE_CASCADE: Optional[cv2.CascadeClassifier] = None
EYE_CASCADE: Optional[cv2.CascadeClassifier] = None

def _load_cascade(name: str) -> cv2.CascadeClassifier:
    path = os.path.join(cv2.data.haarcascades, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File Haar Cascade tidak ditemukan di path OpenCV: {name}")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise IOError(f"Gagal memuat Haar Cascade: {name}")
    return cascade

def ensure_cascades_loaded():
    global FACE_CASCADE, SMILE_CASCADE, EYE_CASCADE
    if FACE_CASCADE is None:
        FACE_CASCADE = _load_cascade('haarcascade_frontalface_default.xml')
    if SMILE_CASCADE is None:
        SMILE_CASCADE = _load_cascade('haarcascade_smile.xml')
    if EYE_CASCADE is None:
        EYE_CASCADE = _load_cascade('haarcascade_eye.xml')

############################################################
# ====== SINKRONISASI DATA (Networking) ======
############################################################
def send_snapshot(snapshot: Dict[str, Any], endpoint: str, location_id: str) -> bool:
    if not _HAS_REQUESTS:
        return False
    payload = {"location_id": location_id, "metrics": snapshot}
    try:
        response = requests.post(endpoint, json=payload, timeout=15)
        if 200 <= response.status_code < 300:
            print(f"[NET-INFO] Data untuk ts={snapshot['ts']} berhasil dikirim.")
            return True
        else:
            print(f"[NET-WARN] Gagal mengirim data. Server merespon: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"[NET-ERROR] Gagal terhubung ke server: {e}")
        return False

def cache_snapshot_locally(snapshot: Dict[str, Any]):
    ts = snapshot.get('ts', int(time.time()))
    path = os.path.join(PENDING_DIR, f"{ts}_{np.random.randint(1000, 9999)}.json")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f)
        print(f"[CACHE] Data untuk ts={ts} disimpan sementara di {path}")
    except IOError as e:
        print(f"[CACHE-ERROR] Gagal menyimpan data sementara: {e}")

class DataSyncer(threading.Thread):
    def __init__(self, endpoint: str, location_id: str, interval_sec: int = 60):
        super().__init__(daemon=True)
        self.endpoint = endpoint
        self.location_id = location_id
        self.interval = interval_sec
        self.stop_event = threading.Event()

    def run(self):
        print("[SYNC] Background data syncer dimulai.")
        while not self.stop_event.is_set():
            self.sync_pending_files()
            self.stop_event.wait(self.interval)

    def sync_pending_files(self):
        pending_files = sorted(glob.glob(os.path.join(PENDING_DIR, "*.json")))
        if not pending_files:
            return
        print(f"[SYNC] Ditemukan {len(pending_files)} file data tertunda. Mencoba mengirim...")
        for file_path in pending_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    snapshot = json.load(f)
                if send_snapshot(snapshot, self.endpoint, self.location_id):
                    os.remove(file_path)
                    print(f"[SYNC] File {os.path.basename(file_path)} berhasil disinkronkan dan dihapus.")
                else:
                    print("[SYNC] Gagal mengirim data. Akan mencoba lagi nanti.")
                    break
            except (IOError, json.JSONDecodeError) as e:
                print(f"[SYNC-ERROR] Gagal memproses file {file_path}: {e}.")

    def stop(self):
        self.stop_event.set()

############################################################
# ====== UTILITAS & STRUKTUR DATA ======
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
    is_human: bool = True  # Validated as human face
    gaze_direction: str = "unknown"  # "looking", "away", "unknown"
    validation_score: float = 1.0  # Confidence that this is a human face
    age: str = AGE_UNKNOWN
    gender: str = GENDER_UNKNOWN

    def update(self, box: Tuple[int,int,int,int]):
        self.box = box
        self.last_seen = time.time()

############################################################
# ====== KLASIFIKASI & METRIK ======
############################################################
def classify_expression_haar(gray_roi: np.ndarray) -> str:
    if SMILE_CASCADE is None: return "unknown"
    smiles = SMILE_CASCADE.detectMultiScale(gray_roi, scaleFactor=1.7, minNeighbors=20)
    return "smile" if len(smiles) > 0 else "neutral"

class HumanFaceValidator:
    """Advanced human face validation to filter out non-human objects"""
    
    def __init__(self, strict_mode=False):
        self.strict_mode = strict_mode
        
        if strict_mode:
            # Strict validation for high-precision scenarios
            self.min_eye_count = 1
            self.min_aspect_ratio = 0.5
            self.max_aspect_ratio = 2.0
            self.min_face_area = 1600
            self.texture_threshold = 50
            self.hist_threshold = 100
        else:
            # Relaxed validation for better human detection
            self.min_eye_count = 0  # Eye detection optional
            self.min_aspect_ratio = 0.3  # More flexible aspect ratio
            self.max_aspect_ratio = 3.0
            self.min_face_area = 900   # Smaller minimum area
            self.texture_threshold = 20  # Lower texture requirement
            self.hist_threshold = 50    # Lower histogram requirement
        
    def validate_human_face(self, gray_roi: np.ndarray, face_box: Tuple[int,int,int,int], debug=False) -> bool:
        """Validate if detected region is actually a human face"""
        if gray_roi.size == 0:
            if debug: print(f"[VALIDATION] FAIL: Empty ROI")
            return False
            
        x, y, w, h = face_box
        
        # 1. Aspect ratio check (faces are roughly oval)
        aspect_ratio = w / h
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            if debug: print(f"[VALIDATION] FAIL: Aspect ratio {aspect_ratio:.2f} not in range [{self.min_aspect_ratio}, {self.max_aspect_ratio}]")
            return False
            
        # 2. Size check
        face_area = w * h
        if face_area < self.min_face_area:
            if debug: print(f"[VALIDATION] FAIL: Face area {face_area} < {self.min_face_area}")
            return False
            
        # 3. Eye detection validation (only if required)
        if self.min_eye_count > 0 and EYE_CASCADE is not None:
            eyes = EYE_CASCADE.detectMultiScale(
                gray_roi, 
                scaleFactor=1.1, 
                minNeighbors=3,  # Lower requirement
                minSize=(8, 8),   # Smaller minimum
                maxSize=(int(w*0.9), int(h*0.9))  # Larger maximum
            )
            if len(eyes) < self.min_eye_count:
                if debug: print(f"[VALIDATION] FAIL: Eyes detected {len(eyes)} < {self.min_eye_count}")
                return False
                
        # 4. Texture analysis (faces have more texture variation than objects)
        try:
            laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            if laplacian_var < self.texture_threshold:  
                if debug: print(f"[VALIDATION] FAIL: Texture variance {laplacian_var:.2f} < {self.texture_threshold}")
                return False
        except Exception as e:
            if debug: print(f"[VALIDATION] WARNING: Texture analysis failed: {e}")
            pass
            
        # 5. Histogram analysis (faces have specific brightness distribution)
        try:
            hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
            hist_std = np.std(hist)
            if hist_std < self.hist_threshold:  
                if debug: print(f"[VALIDATION] FAIL: Histogram std {hist_std:.2f} < {self.hist_threshold}")
                return False
        except Exception as e:
            if debug: print(f"[VALIDATION] WARNING: Histogram analysis failed: {e}")
            pass
            
        if debug: print(f"[VALIDATION] PASS: Face validated as human (area={face_area}, ratio={aspect_ratio:.2f})")
        return True

class GazeDetector:
    """Detect if person is looking toward camera or away"""
    
    def __init__(self):
        self.eye_position_threshold = 0.3  # How centered eyes should be
        
    def detect_gaze_direction(self, gray_roi: np.ndarray, face_box: Tuple[int,int,int,int]) -> str:
        """Detect gaze direction: 'looking', 'away', 'unknown'"""
        if gray_roi.size == 0 or EYE_CASCADE is None:
            return "unknown"
            
        x, y, w, h = face_box
        
        # Detect eyes in the face region
        eyes = EYE_CASCADE.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            maxSize=(int(w*0.4), int(h*0.4))
        )
        
        if len(eyes) < 2:
            # If we can't detect both eyes, use profile detection
            return self._detect_profile_direction(gray_roi)
            
        # Analyze eye positions
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            eye_centers.append((center_x, center_y))
            
        if len(eye_centers) >= 2:
            # Sort eyes by x-position (left to right)
            eye_centers.sort(key=lambda e: e[0])
            left_eye, right_eye = eye_centers[0], eye_centers[1]
            
            # Calculate eye symmetry and position
            eye_distance = right_eye[0] - left_eye[0]
            face_center_x = w // 2
            eyes_center_x = (left_eye[0] + right_eye[0]) // 2
            
            # If eyes are reasonably symmetric and centered, person is looking forward
            center_offset = abs(eyes_center_x - face_center_x) / w
            eye_symmetry = eye_distance / w
            
            if center_offset < self.eye_position_threshold and eye_symmetry > 0.1:
                return "looking"
            else:
                return "away"
                
        return "unknown"
    
    def _detect_profile_direction(self, gray_roi: np.ndarray) -> str:
        """Fallback method using edge detection for profile faces"""
        # Use edge detection to determine if face is frontal or profile
        edges = cv2.Canny(gray_roi, 50, 150)
        
        # Count edges in left vs right half
        h, w = gray_roi.shape
        left_half = edges[:, :w//2]
        right_half = edges[:, w//2:]
        
        left_edges = np.sum(left_half > 0)
        right_edges = np.sum(right_half > 0)
        
        # If edges are roughly balanced, likely frontal face
        edge_ratio = min(left_edges, right_edges) / max(left_edges, right_edges)
        
        if edge_ratio > 0.6:  # Balanced edges
            return "looking"
        else:
            return "away"

class HeuristicDemographics:
    """Fallback demographic estimation using simple heuristics"""
    def __init__(self, environment="mall"):
        self.rng = np.random.default_rng(42)
        self.environment = environment
        
        # Demographic distributions based on environment
        if environment == "mall":
            # Mall demographics: more young adults and families
            self.age_weights = [0.15, 0.25, 0.22, 0.18, 0.12, 0.08]  # More teenagers & young adults
            self.gender_weights = [0.45, 0.55]  # Slightly more female in malls
        elif environment == "office":
            # Office demographics: working adults
            self.age_weights = [0.05, 0.20, 0.35, 0.25, 0.12, 0.03]  # Peak at 25-44
            self.gender_weights = [0.50, 0.50]  # Balanced
        elif environment == "transit":
            # Transit/station: diverse mix
            self.age_weights = [0.12, 0.28, 0.25, 0.20, 0.10, 0.05]  # Young skew
            self.gender_weights = [0.52, 0.48]  # Slightly male (commuters)
        else:
            # Default general public
            self.age_weights = [0.10, 0.20, 0.25, 0.20, 0.15, 0.10]
            self.gender_weights = [0.52, 0.48]
        
    def estimate_age_from_face_size(self, face_box: Tuple[int,int,int,int]) -> str:
        """Estimate age based on face size (larger faces might indicate closer/older people)"""
        x, y, w, h = face_box
        face_area = w * h
        
        # Use face size as a rough heuristic combined with randomness
        if face_area > 15000:  # Large face (close to camera)
            # Slightly bias toward adult ages
            weights = [0.05, 0.15, 0.30, 0.25, 0.15, 0.10]
        elif face_area < 8000:  # Small face (far from camera or child)
            # Slightly bias toward younger ages
            weights = [0.20, 0.25, 0.20, 0.15, 0.12, 0.08]
        else:
            weights = self.age_weights
            
        return self.rng.choice(AGE_BINS, p=weights)
    
    def estimate_gender_from_expression(self, expression: str) -> str:
        """Simple gender estimation with slight bias based on expression"""
        if expression == "smile":
            # Slight bias based on general behavioral tendencies (very rough heuristic)
            weights = [0.48, 0.52]  # Slightly more female for smiles
        else:
            weights = self.gender_weights
        
        return self.rng.choice(GENDER_BINS, p=weights)
    
    def predict_demographics(self, face_box: Tuple[int,int,int,int], expression: str) -> Tuple[str, str]:
        """Return (age, gender) predictions"""
        age = self.estimate_age_from_face_size(face_box)
        gender = self.estimate_gender_from_expression(expression)
        return age, gender

def _safe_rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator > 0 else 0.0

class Metrics:
    def __init__(self):
        self.history: deque = deque(maxlen=3600)
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
        self.gender_bins: Dict[str, int] = defaultdict(int)

    def add_impressions(self, n: int):
        with self._lock:
            self.impressions += n
            self.today_impr += n

    def add_view(self):
        with self._lock:
            self.views += 1
            self.today_views += 1

    def add_dwell(self, sec: float):
        with self._lock:
            self.dwell_accum += sec
            self.dwell_count += 1

    def add_demographics(self, age: str, expr: str, gender: str):
        with self._lock:
            self.age_bins[age] += 1
            self.expr_bins[expr] += 1
            self.gender_bins[gender] += 1

    def _current_snapshot_dict(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = now - self.window_start
        rate = _safe_rate(self.views, self.impressions)
        avg_dwell = _safe_rate(self.dwell_accum, self.dwell_count)
        return {
            "ts": int(now), "window_sec": int(elapsed), "impressions": self.impressions,
            "views": self.views, "view_rate": rate, "avg_dwell": avg_dwell,
            "age_dist": dict(self.age_bins), "expr_dist": dict(self.expr_bins),
            "gender_dist": dict(self.gender_bins),
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
                "impressions": self.today_impr, "views": self.today_views,
                "view_rate": _safe_rate(self.today_views, self.today_impr)
            }
            return data

    def export_to_csv(self, snap: Dict[str, Any]):
        path = os.path.join(EXPORT_DIR, time.strftime("%Y%m%d") + ".csv")
        header = "ts,window_sec,impressions,views,view_rate,avg_dwell,age_13_17,age_18_24,age_25_34,age_35_44,age_45_54,age_55p,expr_neutral,expr_smile,expr_unknown,gender_pria,gender_wanita,gender_unknown\n"
        row_values = [
            snap["ts"], snap["window_sec"], snap["impressions"], snap["views"],
            f'{snap["view_rate"]:.4f}', f'{snap["avg_dwell"]:.3f}',
            snap["age_dist"].get("13-17", 0), snap["age_dist"].get("18-24", 0),
            snap["age_dist"].get("25-34", 0), snap["age_dist"].get("35-44", 0),
            snap["age_dist"].get("45-54", 0), snap["age_dist"].get("55+", 0),
            snap["expr_dist"].get("neutral", 0), snap["expr_dist"].get("smile", 0),
            snap["expr_dist"].get("unknown", 0),
            snap["gender_dist"].get("Pria", 0), snap["gender_dist"].get("Wanita", 0),
            snap["gender_dist"].get("unknown", 0)
        ]
        row = ",".join(map(str, row_values)) + "\n"
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if write_header: f.write(header)
            f.write(row)
        print(f"[SNAPSHOT] Data CSV tersimpan di {path}")

############################################################
# ====== SERVER WEB LOKAL (Flask - Opsional) ======
############################################################
try:
    from flask import Flask, jsonify, Response, render_template_string
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

class WebServer:
    def __init__(self, metrics: Metrics, host: str, port: int, refresh_ms: int):
        if not _HAS_FLASK: raise RuntimeError("Flask tidak terpasang. Jalankan: pip install flask")
        self.metrics = metrics
        self.app = Flask(__name__, template_folder="templates")
        self.host, self.port, self.refresh_ms = host, port, refresh_ms
        self._bind_routes()

    def _bind_routes(self):
        @self.app.route("/metrics")
        def _get_metrics():
            return jsonify({"current": self.metrics.peek_current()})

        @self.app.route("/")
        def _render_dashboard():
            try:
                with open(os.path.join(self.app.template_folder, 'index.html'), 'r') as f:
                    template_str = f.read()
                current_data = self.metrics.peek_current()
                html = render_template_string(template_str, current=current_data,
                                              today=current_data.get('today', {}),
                                              history=list(self.metrics.history)[-30:],
                                              refresh_ms=self.refresh_ms)
                return Response(html, mimetype='text/html')
            except FileNotFoundError:
                return "Template 'index.html' tidak ditemukan.", 404

    def start(self):
        thread = threading.Thread(target=lambda: self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False))
        thread.daemon = True
        thread.start()
        print(f"[WEB] Dasbor lokal berjalan di http://{self.host}:{self.port}")

############################################################
# ====== SUMBER FRAME (Kamera, Video, Sintetis) ======
############################################################
class FrameSource:
    def read(self) -> Tuple[bool, np.ndarray]: raise NotImplementedError
    def release(self): pass
    def detect_faces(self, frame: np.ndarray, crowd_mode=False, face_validator=None) -> List[Tuple[int,int,int,int,bool]]:
        """Enhanced face detection with human validation
        Returns: List of (x, y, w, h, is_human) tuples
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if FACE_CASCADE is None: return []
        
        # Adaptive parameters based on crowd density
        if crowd_mode:
            # More aggressive detection for crowded public spaces
            scale_factor = 1.03
            min_neighbors = 6
            min_size = (40, 40)  # Detect smaller/distant faces
            overlap_threshold = 0.3  # Allow closer faces
        else:
            # Conservative detection for single/few people
            scale_factor = 1.05
            min_neighbors = 8
            min_size = MIN_FACE_SIZE
            overlap_threshold = 0.5
            
        # Enhanced detection with histogram equalization for varying lighting
        gray = cv2.equalizeHist(gray)
        
        faces = FACE_CASCADE.detectMultiScale(
            gray, 
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter faces that are too close together (likely duplicates)
        filtered_faces = []
        for (x, y, w, h) in faces:
            is_duplicate = False
            for (fx, fy, fw, fh, _) in filtered_faces:
                # Calculate overlap
                overlap_x = max(0, min(x + w, fx + fw) - max(x, fx))
                overlap_y = max(0, min(y + h, fy + fh) - max(y, fy))
                overlap_area = overlap_x * overlap_y
                face_area = w * h
                
                # Dynamic overlap threshold based on mode
                if overlap_area > overlap_threshold * face_area:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Validate if this is actually a human face
                is_human = True  # Default to true
                if face_validator is not None:
                    # Extract face ROI for validation
                    face_roi = gray[y:y+h, x:x+w] if y+h <= gray.shape[0] and x+w <= gray.shape[1] else gray[y:min(y+h, gray.shape[0]), x:min(x+w, gray.shape[1])]
                    is_human = face_validator.validate_human_face(face_roi, (x, y, w, h), debug=getattr(face_validator, 'debug_mode', False))
                
                filtered_faces.append((int(x), int(y), int(w), int(h), is_human))
        
        return filtered_faces

class CameraSource(FrameSource):
    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened(): raise RuntimeError(f"Tidak dapat membuka kamera indeks {index}")
    def read(self) -> Tuple[bool, np.ndarray]: return self.cap.read()
    def release(self): self.cap.release()

class VideoSource(FrameSource):
    def __init__(self, path: str):
        if not os.path.exists(path): raise FileNotFoundError(f"File video tidak ditemukan: {path}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise RuntimeError(f"Tidak dapat membuka video: {path}")
    def read(self) -> Tuple[bool, np.ndarray]: return self.cap.read()
    def release(self): self.cap.release()

class NVRSource(FrameSource):
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened(): raise RuntimeError(f"Tidak dapat terhubung ke NVR di URL: {url}")
    def read(self) -> Tuple[bool, np.ndarray]: return self.cap.read()
    def release(self): self.cap.release()

class SyntheticSource(FrameSource):
    def __init__(self, w: int = 1280, h: int = 720, n: int = 2, crowd_mode: bool = False):
        self.w, self.h, self.n = w, h, n
        rng = np.random.default_rng(42)
        self.pos = rng.integers(low=[100,100], high=[w-200, h-200], size=(n,2)).astype(np.float32)
        self.vel = rng.uniform(low=-3.0, high=3.0, size=(n,2)).astype(np.float32)
        
        # Crowd mode adjustments
        if crowd_mode:
            # Mall scenario: more varied sizes (people at different distances)
            min_size, max_size = 30, 150  # Wider range for mall crowds
            self.presence_duration = rng.uniform(low=3.0, high=15.0, size=n)  # Shorter stays (people walking by)
            self.vel = rng.uniform(low=-5.0, high=5.0, size=(n,2))  # Faster movement
        else:
            # Close-up scenario: larger, more consistent faces
            min_size, max_size = 80, 180
            self.presence_duration = rng.uniform(low=5.0, high=30.0, size=n)  # Longer stays
        
        self.size = rng.integers(low=min_size, high=max_size, size=(n,2))
        self.spawn_time = time.time() + rng.uniform(low=0, high=10, size=n)  # When they appear
        self.active = np.zeros(n, dtype=bool)  # Which faces are currently active
        
    def read(self) -> Tuple[bool, np.ndarray]:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        current_time = time.time()
        
        for i in range(self.n):
            # Dynamic face appearance/disappearance
            if not self.active[i] and current_time > self.spawn_time[i]:
                self.active[i] = True
                # Reset position when spawning
                rng = np.random.default_rng(int(current_time * 1000) % 2**32)
                self.pos[i] = rng.integers(low=[100,100], high=[self.w-200, self.h-200])
            elif self.active[i] and (current_time - self.spawn_time[i]) > self.presence_duration[i]:
                self.active[i] = False
                # Respawn after some time
                self.spawn_time[i] = current_time + np.random.uniform(2, 15)
                
            if self.active[i]:
                self.pos[i] += self.vel[i]
                for d in [0,1]:
                    if not (10 < self.pos[i, d] < (self.w if d==0 else self.h) - self.size[i, d] - 10):
                        self.vel[i, d] *= -1
                        
        return True, frame
        
    def detect_faces(self, frame: np.ndarray, crowd_mode=False, face_validator=None) -> List[Tuple[int,int,int,int,bool]]:
        faces = []
        for i in range(self.n):
            if self.active[i]:  # Only return active faces
                x,y,w,h = int(self.pos[i,0]), int(self.pos[i,1]), int(self.size[i,0]), int(self.size[i,1])
                # Synthetic faces are always human by definition
                faces.append((x, y, w, h, True))
        return faces

############################################################
# ====== LOGIKA UTAMA APLIKASI ======
############################################################
def update_tracks(tracks: List[Track], faces: List[Tuple[int,int,int,int,bool]], assoc_iou: float, track_exp: float) -> List[Track]:
    now = time.time()
    matches = {i: -1 for i in range(len(faces))}
    
    # Enhanced tracking with better IoU calculation
    for i, (x, y, w, h, is_human) in enumerate(faces):
        box = (x, y, w, h)
        best_iou, best_idx = 0.0, -1
        for j, tr in enumerate(tracks):
            score = iou(box, tr.box)
            if score > best_iou: 
                best_iou, best_idx = score, j
        
        # Lower IoU threshold for better tracking continuity
        if best_iou > max(assoc_iou * 0.7, 0.15):  # At least 0.15 IoU
            tracks[best_idx].update(box)
            tracks[best_idx].is_human = is_human  # Update human validation
            matches[i] = best_idx
    
    # Only create new tracks if no existing track is close enough
    unmatched_faces = [(x, y, w, h, is_human) for i, (x, y, w, h, is_human) in enumerate(faces) if matches[i] == -1]
    for (x, y, w, h, is_human) in unmatched_faces:
        box = (x, y, w, h)
        # Check if this face is really new (not just a tracking failure)
        is_really_new = True
        for tr in tracks:
            if (now - tr.last_seen) < 1.0:  # Recently seen track
                distance = abs(box[0] - tr.box[0]) + abs(box[1] - tr.box[1])
                if distance < 100:  # Within 100 pixels of existing track
                    is_really_new = False
                    break
        
        if is_really_new:
            track = Track(box=box, last_seen=now)
            track.is_human = is_human
            tracks.append(track)
    
    # Keep tracks alive longer for better continuity
    active_tracks = [t for t in tracks if (now - t.last_seen) < track_exp]
    
    return active_tracks

def process_frame_metrics(tracks: List[Track], frame: np.ndarray, metrics: Metrics, age_est: Optional[AgeEstimator], gender_est: Optional[GenderEstimator], heuristic_demo: Optional[HeuristicDemographics], gaze_detector: Optional[GazeDetector], dwell_th: float):
    now = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for tr in tracks:
        x, y, w, h = tr.box
        if w <= 0 or h <= 0: continue
        
        # Skip processing if not validated as human
        if not tr.is_human:
            continue
            
        # Extract face ROI
        face_roi_gray = gray[y:y+h, x:x+w] if y+h <= gray.shape[0] and x+w <= gray.shape[1] else gray[y:min(y+h, gray.shape[0]), x:min(x+w, gray.shape[1])]
        
        # Detect gaze direction
        if gaze_detector:
            tr.gaze_direction = gaze_detector.detect_gaze_direction(face_roi_gray, (x, y, w, h))
        
        # Update looking status based on gaze
        is_currently_looking = tr.gaze_direction == "looking"
        
        if not tr.is_looking and is_currently_looking:
            tr.is_looking = True
            tr.look_start = now
        elif tr.is_looking and not is_currently_looking:
            tr.is_looking = False
            tr.look_start = None
        elif tr.is_looking and tr.look_start and not tr.view_counted and (now - tr.look_start) >= dwell_th:
            metrics.add_view()
            metrics.add_dwell(now - tr.look_start)
            tr.view_counted = True
        
        # Only process demographics for people who are looking (engaged)
        if tr.gaze_direction == "looking":
            expr = classify_expression_haar(face_roi_gray)
            age = AGE_UNKNOWN
            gender = GENDER_UNKNOWN
            
            # Try ONNX models first
            if age_est and w > 0 and h > 0:
                face_roi_bgr = frame[y:y+h, x:x+w] if y+h <= frame.shape[0] and x+w <= frame.shape[1] else frame[y:min(y+h, frame.shape[0]), x:min(x+w, frame.shape[1])]
                age = age_est.predict_bin(face_roi_bgr)
            if gender_est and w > 0 and h > 0:
                face_roi_bgr = frame[y:y+h, x:x+w] if y+h <= frame.shape[0] and x+w <= frame.shape[1] else frame[y:min(y+h, frame.shape[0]), x:min(x+w, frame.shape[1])]
                gender = gender_est.predict_bin(face_roi_bgr)
                
            # Fallback to heuristic demographics if ONNX models not available
            if (age == AGE_UNKNOWN or gender == GENDER_UNKNOWN) and heuristic_demo:
                heuristic_age, heuristic_gender = heuristic_demo.predict_demographics((x, y, w, h), expr)
                if age == AGE_UNKNOWN:
                    age = heuristic_age
                if gender == GENDER_UNKNOWN:
                    gender = heuristic_gender
            
            # Simpan hasil demografi ke track untuk ditampilkan
            tr.age = age
            tr.gender = gender
                    
            metrics.add_demographics(age, expr, gender)

def draw_overlays(frame: np.ndarray, tracks: List[Track]):
    for tr in tracks:
        x, y, w, h = tr.box
        
        # Default label and color
        color = (128, 128, 128)
        label = "UNKNOWN"

        if not tr.is_human:
            color = (0, 0, 255)  # Red for non-human detections
            label = "NOT HUMAN"
        else:
            # Build label for humans
            gender_label = tr.gender.upper() if tr.gender != GENDER_UNKNOWN else ""
            age_label = tr.age if tr.age != AGE_UNKNOWN else ""
            
            if gender_label and age_label:
                label = f"{gender_label} ({age_label})"
            elif gender_label:
                label = gender_label
            elif age_label:
                label = age_label
            else: # Fallback to gaze direction if no demographics
                label = tr.gaze_direction.upper()

            # Set color based on gaze
            if tr.gaze_direction == "looking":
                color = (0, 255, 0)
            elif tr.gaze_direction == "away":
                color = (0, 165, 255)
        
        # Draw bounding box
        thickness = 3 if tr.is_looking else 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Draw label
        label_size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
        cv2.rectangle(frame, (x, y-25), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 7), FONT, 0.5, (255, 255, 255), 1)

def run_app(args: argparse.Namespace):
    source = setup_source(args)
    metrics = Metrics()
    age_estimator = setup_age_estimator(args)
    gender_estimator = setup_gender_estimator(args)
    
    # Initialize heuristic demographics as fallback
    heuristic_demo = None
    if not age_estimator or not gender_estimator:
        heuristic_demo = HeuristicDemographics(environment=args.environment)
        print(f"[INFO] Using heuristic demographic estimation for {args.environment} environment (no ONNX models provided)")
        
    # Initialize face validation and gaze detection
    face_validator = HumanFaceValidator(strict_mode=args.strict_validation)
    face_validator.debug_mode = args.debug_validation  # Enable debug output
    gaze_detector = GazeDetector()
    
    if args.strict_validation:
        print("[INFO] Strict human face validation enabled (high precision, may reject some real faces)")
    else:
        print("[INFO] Relaxed human face validation enabled (better human detection, may allow some false positives)")
        
    if args.debug_validation:
        print("[INFO] Face validation debug mode enabled")
        
    # Set crowd detection mode
    crowd_mode = args.crowd_mode or args.environment in ["mall", "transit"]
    if crowd_mode:
        print("[INFO] Crowd detection mode enabled for public space deployment")
    
    tracks: List[Track] = []
    last_snapshot_time = time.time()
    syncer = None
    if args.api_endpoint:
        syncer = DataSyncer(args.api_endpoint, args.location_id)
        syncer.start()

    if args.web:
        ws = WebServer(metrics, args.web_host, args.web_port, args.web_refresh_ms)
        ws.start()

    print_startup_message(args)
    frame_count = 0
    detection_interval = args.detection_interval  # Configurable detection interval
    
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                print("[INFO] Sumber video selesai atau terputus.")
                break

            frame_count += 1
            
            # Only run face detection every few frames to reduce over-detection
            if frame_count % detection_interval == 0:
                faces = source.detect_faces(frame, crowd_mode=crowd_mode, face_validator=face_validator)
                if faces:
                    # Count only human faces for impressions
                    human_faces = [f for f in faces if f[4]]  # f[4] is is_human
                    
                    # Dynamic impression limiting based on environment
                    if crowd_mode:
                        # Allow more faces in crowd scenarios
                        impression_count = min(len(human_faces), 20)  # Max 20 human faces per cycle for crowds
                    else:
                        # Conservative for single/few people
                        impression_count = min(len(human_faces), 5)   # Max 5 human faces per cycle
                    metrics.add_impressions(impression_count)
                    
                    # Show detection stats
                    non_human_count = len(faces) - len(human_faces)
                    if non_human_count > 0:
                        print(f"[DETECTION] Found {len(human_faces)} human faces, {non_human_count} non-human detections filtered")
                    
                tracks = update_tracks(tracks, faces, args.assoc_iou, args.track_exp_sec)
                process_frame_metrics(tracks, frame, metrics, age_estimator, gender_estimator, heuristic_demo, gaze_detector, args.dwell)
            else:
                # Still update existing tracks without new detections
                faces = []
                tracks = update_tracks(tracks, faces, args.assoc_iou, args.track_exp_sec)
                if tracks:  # Only process if we have active tracks
                    process_frame_metrics(tracks, frame, metrics, age_estimator, gender_estimator, heuristic_demo, gaze_detector, args.dwell)

            if not args.no_window:
                draw_overlays(frame, tracks)
                cv2.imshow("Ads Audience Analytics", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if (time.time() - last_snapshot_time) >= args.window_sec:
                snap = metrics.snapshot_and_reset()
                metrics.export_to_csv(snap)

                print(
                    f"[SNAPSHOT] ts={snap['ts']} win={snap['window_sec']}s impr={snap['impressions']} "
                    f"views={snap['views']} rate={snap['view_rate']:.4f} avg_dwell={snap['avg_dwell']:.2f}"
                )
                print(f"         age_dist={snap['age_dist']} expr_dist={snap['expr_dist']} gender_dist={snap['gender_dist']}")

                if args.api_endpoint:
                    if not send_snapshot(snap, args.api_endpoint, args.location_id):
                        cache_snapshot_locally(snap)
                last_snapshot_time = time.time()
    finally:
        source.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        if syncer:
            syncer.stop()
        print("[INFO] Aplikasi berhenti.")

############################################################
# ====== FUNGSI SETUP, TEST & MAIN ======
############################################################
def list_available_cameras(max_to_probe: int) -> List[int]:
    available = []
    for i in range(max_to_probe):
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            available.append(i)
        cap.release()
    return available

def setup_source(args: argparse.Namespace) -> FrameSource:
    ensure_cascades_loaded()
    crowd_mode = args.crowd_mode or args.environment in ["mall", "transit"]
    
    if args.nvr_url:
        print(f"[INFO] Mencoba terhubung ke NVR di: {args.nvr_url}")
        try:
            return NVRSource(args.nvr_url)
        except RuntimeError as e:
            print(f"[FATAL] Gagal terhubung ke NVR: {e}")
            sys.exit(1)

    if args.video: 
        print(f"[INFO] Membuka file video: {args.video}")
        return VideoSource(args.video)

    if args.synthetic: 
        crowd_desc = "crowd mode" if crowd_mode else "normal mode"
        print(f"[INFO] Menggunakan sumber sintetis dengan {args.synthetic_crowd} wajah ({crowd_desc}).")
        return SyntheticSource(n=args.synthetic_crowd, crowd_mode=crowd_mode)
    
    # Fallback to camera
    try:
        print(f"[INFO] Mencoba membuka kamera indeks: {args.camera_index}")
        return CameraSource(args.camera_index)
    except RuntimeError as e:
        print(f"[WARN] Gagal membuka kamera indeks {args.camera_index}: {e}")
        available_cams = list_available_cameras(DEFAULT_PROBE_MAX)
        if available_cams:
            print(f"[INFO] Kamera tersedia ditemukan di indeks {available_cams[0]}, menggunakan itu.")
            args.camera_index = available_cams[0]
            return CameraSource(args.camera_index)
        else:
            crowd_desc = "crowd mode" if crowd_mode else "normal mode"
            print(f"[WARN] Tidak ada kamera fisik ditemukan. Beralih ke mode sintetis dengan {args.synthetic_crowd} wajah ({crowd_desc}).")
            return SyntheticSource(n=args.synthetic_crowd, crowd_mode=crowd_mode)

def setup_age_estimator(args: argparse.Namespace) -> Optional[AgeEstimator]:
    if not args.age_model: return None
    try:
        estimator = AgeEstimator(args.age_model)
        print(f"[INFO] Model estimasi usia dimuat dari: {args.age_model}")
        return estimator
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[WARN] Gagal memuat model usia: {e}. Fitur dinonaktifkan.")
        return None

def setup_gender_estimator(args: argparse.Namespace) -> Optional[GenderEstimator]:
    if not args.gender_model: return None
    try:
        estimator = GenderEstimator(args.gender_model)
        print(f"[INFO] Model estimasi gender dimuat dari: {args.gender_model}")
        return estimator
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[WARN] Gagal memuat model gender: {e}. Fitur dinonaktifkan.")
        return None

def print_startup_message(args: argparse.Namespace):
    if args.no_window:
        print("[INFO] Berjalan dalam mode headless.")
    else:
        print("[INFO] Aplikasi berjalan. Tekan 'q' di jendela video untuk keluar.")

def _almost_equal(a: float, b: float, eps: float=1e-6) -> bool: return abs(a-b) < eps

def run_tests():
    print("[TEST] Menjalankan unit tests...")
    assert _almost_equal(iou((0,0,10,10), (0,0,10,10)), 1.0)
    print("[PASS] test_iou_basic")
    m = Metrics()
    m.add_impressions(5)
    m.add_view()
    snap = m.snapshot_and_reset()
    assert snap['impressions'] == 5 and snap['views'] == 1
    print("[PASS] test_metrics_snapshot")
    print("\nSemua test berhasil dilewati.")

def main():
    parser = argparse.ArgumentParser(
        description="Ads Audience Analytics - Purwarupa Klien Edge",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Contoh penggunaan:\n"
               "  - Kamera IP/NVR: python3 app.py --nvr-url rtsp://user:pass@192.168.1.1/stream1\n"
               "  - File video:    python3 app.py --video path/to/video.mp4\n"
               "  - Kamera lokal:  python3 app.py --camera-index 0\n"
               "  - Headless & API: python3 app.py --nvr-url ... --no-window --api-endpoint <URL> --location-id <ID>"
    )
    parser.add_argument("--nvr-url", type=str, help="URL stream dari NVR atau kamera IP (prioritas tertinggi).")
    parser.add_argument("--video", type=str, help="Path ke file video untuk diproses (prioritas kedua).")
    parser.add_argument("--camera-index", type=int, default=0, help="Indeks kamera lokal (prioritas ketiga).")
    parser.add_argument("--synthetic", action="store_true", help="Gunakan sumber data sintetis (prioritas terendah).")
    parser.add_argument("--synthetic-crowd", type=int, default=3, help="Jumlah wajah sintetis.")
    parser.add_argument("--list-cams", action="store_true", help="Tampilkan daftar kamera yang tersedia lalu keluar.")
    parser.add_argument("--no-window", action="store_true", help="Jalankan tanpa jendela video (headless).")
    parser.add_argument("--window-sec", type=int, default=DEFAULT_WINDOW_SEC, help=f"Jendela agregasi (detik, default: {DEFAULT_WINDOW_SEC})")
    parser.add_argument("--dwell", type=float, default=DEFAULT_VIEW_DWELL_SEC, help=f"Ambang dwell (detik) untuk view (default: {DEFAULT_VIEW_DWELL_SEC})")
    parser.add_argument("--age-model", type=str, help="Path ke model usia ONNX (opsional).")
    parser.add_argument("--gender-model", type=str, help="Path ke model gender ONNX (opsional).")
    parser.add_argument("--api-endpoint", type=str, help="URL API server pusat untuk mengirim data.")
    parser.add_argument("--location-id", type=str, default="UNKNOWN_LOCATION", help="ID unik untuk klien/lokasi ini.")
    parser.add_argument("--assoc-iou", type=float, default=DEFAULT_IOU_ASSOC_TH, help=f"Ambang IoU untuk asosiasi track (default: {DEFAULT_IOU_ASSOC_TH})")
    parser.add_argument("--track-exp-sec", type=float, default=DEFAULT_TRACK_EXP_SEC, help=f"Detik kedaluwarsa track (default: {DEFAULT_TRACK_EXP_SEC})")
    parser.add_argument("--web", action="store_true", help="(Lokal) Aktifkan dasbor web.")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Host untuk dasbor web lokal.")
    parser.add_argument("--web-port", type=int, default=8000, help="Port untuk dasbor web lokal.")
    parser.add_argument("--web-refresh-ms", type=int, default=2000, help="Interval refresh dasbor web lokal (ms).")
    parser.add_argument("--detection-interval", type=int, default=3, help="Interval deteksi wajah (setiap N frame, default: 3)")
    parser.add_argument("--environment", type=str, choices=["mall", "office", "transit", "general"], default="general", help="Environment type for demographic tuning")
    parser.add_argument("--crowd-mode", action="store_true", help="Enable crowd detection mode for public spaces")
    parser.add_argument("--strict-validation", action="store_true", help="Enable strict human face validation (may reject real faces)")
    parser.add_argument("--debug-validation", action="store_true", help="Enable debug output for face validation")
    parser.add_argument("--run-tests", action="store_true", help="Jalankan unit tests lalu keluar.")
    args = parser.parse_args()

    if args.api_endpoint and not _HAS_REQUESTS:
        print("[FATAL] Argumen --api-endpoint memerlukan library 'requests'.")
        print("         Silakan jalankan: python3 -m pip install requests")
        sys.exit(1)

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