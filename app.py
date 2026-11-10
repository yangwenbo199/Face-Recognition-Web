# -*- coding: utf-8 -*-
"""人脸识别 Web 系统 - Flask 主程序"""
import os
import json
import time
import threading
import functools
import numpy as np
import cv2
from flask import Flask, render_template, Response, request, jsonify
from insightface.app import FaceAnalysis
import face_db


import sys

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = r"D:\face_models"
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


# ---------- 配置 ----------
DEFAULT_CONFIG = {"TOL": 0.50, "cam_id": 0, "det_size": 320, "show_dist": True}
API_TOKEN = "YWB2LH"          # 改成你自己的随机串
PHOTO_TOL = 0.45              # 照片查询专用阈值（余弦距离，越小越像）


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(json.load(f))
            return cfg
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


CFG = load_config()


# ---------- 鉴权装饰器 ----------
def require_token(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if request.headers.get("X-Token") != API_TOKEN:
            return jsonify({"ok": False, "msg": "未授权"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ---------- 人脸工具 ----------
def pick_largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def check_face_quality(face, min_size=80, max_angle=30):
    """返回 (True, '') 或 (False, 原因)"""
    w = face.bbox[2] - face.bbox[0]
    h = face.bbox[3] - face.bbox[1]
    if min(w, h) < min_size:
        return False, "人脸太小，请上传更清晰的正脸照"
    if hasattr(face, "pose") and face.pose is not None:
        pitch, yaw = float(face.pose[0]), float(face.pose[1])
        if abs(yaw) > max_angle or abs(pitch) > max_angle:
            return False, "人脸角度过大，请使用正脸照"
    return True, ""


# ---------- 模型 ----------
_app = None


def get_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(
            name="buffalo_l",
            root=MODEL_ROOT,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        _app.prepare(ctx_id=-1, det_size=(CFG["det_size"], CFG["det_size"]))
    return _app


# ---------- 识别核心 ----------
def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(1.0 - np.dot(a, b))


def match(feature, db, tol):
    if not db:
        return "未知", 9.9
    best_name, best_dist = None, 9.9
    for name, feat in db.items():
        d = cosine_distance(feature, feat)
        if d < best_dist:
            best_name, best_dist = name, d
    return (best_name if best_dist <= tol else "未知"), best_dist


# ---------- 状态 ----------
_last_result = {"name": "—", "dist": 0.0, "fps": 0.0, "ts": ""}
_last_lock = threading.Lock()


def gen_frames():
    app = get_app()
    cap = cv2.VideoCapture(CFG["cam_id"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] 摄像头打不开")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    t_prev = time.time()
    frame_count = 0
    last_log_ts = 0.0

    DET_INTERVAL = 3
    loop_i = 0
    cached_faces = []
    cached_db = {}
    db_refresh_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        loop_i += 1
        now = time.time()

        if now - db_refresh_ts > 1.0:
            cached_db = face_db.load_db()
            db_refresh_ts = now

        if loop_i % DET_INTERVAL == 0 or not cached_faces:
            tol = CFG["TOL"]
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                small = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
                small = frame

            faces = app.get(small)
            cached_faces = []

            for f in faces:
                emb = f.embedding
                name, dist = match(emb, cached_db, tol)
                x1, y1, x2, y2 = (f.bbox / scale).astype(int)
                cached_faces.append(((x1, y1, x2, y2), name, dist))

            if now - last_log_ts > 0.5 and cached_faces:
                best = min(cached_faces, key=lambda x: x[2])
                with _last_lock:
                    _last_result["name"] = best[1]
                    _last_result["dist"] = round(best[2], 4)
                    _last_result["ts"] = time.strftime("%H:%M:%S")
                last_log_ts = now

        for (x1, y1, x2, y2), name, dist in cached_faces:
            color = (0, 255, 0) if name != "未知" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = name
            if CFG["show_dist"]:
                label += f" {dist:.3f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_count += 1
        if frame_count >= 10:
            t_now = time.time()
            fps = frame_count / (t_now - t_prev + 1e-6)
            t_prev = t_now
            frame_count = 0
            with _last_lock:
                _last_result["fps"] = round(fps, 1)

        ret, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

    cap.release()


# ---------- Flask ----------
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


@app.route("/")
def page_index():
    return render_template("index.html")


@app.route("/recognize")
def page_recognize():
    return render_template("recognize.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with _last_lock:
        data = dict(_last_result)
    data["people_count"] = len(face_db.load_db())
    data["tol"] = CFG["TOL"]
    return jsonify(data)


@app.route("/api/people", methods=["GET"])
def api_people_list():
    return jsonify(face_db.list_people())


@app.route("/api/people/<name>", methods=["DELETE"])
def api_people_delete(name):
    if not face_db.delete_person(name):
        return jsonify({"ok": False, "msg": "人员不存在"}), 404
    return jsonify({"ok": True})


@app.route("/api/enroll", methods=["POST"])
@require_token
def api_enroll():
    name = request.form.get("name", "").strip()
    contact = request.form.get("contact", "").strip()
    # 前端复选框传 "1" / "true" / "on" 都算同意
    consent = request.form.get("consent", "").lower() in ("1", "true", "on", "yes")

    if not name:
        return jsonify({"ok": False, "msg": "姓名不能为空"}), 400
    if "file" not in request.files:
        return jsonify({"ok": False, "msg": "请上传照片"}), 400

    file = request.files["file"]
    raw = file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "msg": "图片解码失败"}), 400

    face_app = get_app()
    faces = face_app.get(img)
    if not faces:
        return jsonify({"ok": False, "msg": "图中未检测到人脸"}), 400

    largest = pick_largest_face(faces)
    ok, err = check_face_quality(largest)
    if not ok:
        return jsonify({"ok": False, "msg": err}), 400

    face_db.add_person(name, largest.embedding, contact=contact)

    # 本人勾选了同意 → 写一条授权记录
    if consent:
        try:
            face_db.add_consent(name, purpose="联系方式共享", remark="录入页面本人勾选")
        except Exception as e:
            print(f"[!] 写入 consents 失败：{e}")

    return jsonify({"ok": True, "msg": f"已录入 {name}"})


@app.route("/api/recognize_photo", methods=["POST"])
@require_token
def api_recognize_photo():
    if "file" not in request.files:
        return jsonify({"ok": False, "msg": "请上传照片"}), 400

    raw = request.files["file"].read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "msg": "图片解码失败"}), 400

    faces = get_app().get(img)
    if not faces:
        return jsonify({"ok": False, "msg": "图中未检测到人脸"}), 400

    largest = pick_largest_face(faces)
    ok, err = check_face_quality(largest)
    if not ok:
        return jsonify({"ok": False, "msg": err}), 400

    db = face_db.load_db()
    name, dist = match(largest.embedding, db, PHOTO_TOL)

    if name == "未知":
        return jsonify({
            "ok": True, "matched": False,
            "dist": round(dist, 4),
            "msg": "库中无匹配人员"
        })

    person = face_db.get_person(name) or {}

    # 只有本人同意过，才返回联系方式
    has_consent = False
    try:
        has_consent = face_db.has_consent(name, purpose="联系方式共享")
    except Exception:
        has_consent = False

    return jsonify({
        "ok": True,
        "matched": True,
        "name": name,
        "contact": person.get("contact", "") if has_consent else "",
        "contact_allowed": has_consent,
        "dist": round(dist, 4)
    })


if __name__ == "__main__":
    print("启动 Web 服务：http://127.0.0.1:5000")
    print(f"当前配置：{CFG}")
    print(">>> 预加载模型，请稍候…")
    _t0 = time.time()
    get_app()
    print(f">>> 模型加载完成，耗时 {time.time() - _t0:.1f} 秒")
        # 延迟1.5秒后自动打开浏览器
    import webbrowser
    import threading
    threading.Timer(1.5, lambda: webbrowser.open_new("http://127.0.0.1:5000")).start()

    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)

