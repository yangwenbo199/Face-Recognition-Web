# -*- coding: utf-8 -*-
"""人脸数据库 - SQLite 版（含联系方式 + 授权留痕）"""
import os
import sqlite3
import threading
import numpy as np

import sys

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "face_db.db")

_lock = threading.RLock()


def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    with _lock, _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS person (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL UNIQUE,
                feature    BLOB    NOT NULL,
                contact    TEXT    DEFAULT '',
                consent_at TEXT    DEFAULT '',
                consent_scope TEXT DEFAULT '',
                created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_person_name ON person(name)")

        # 老库兼容：缺字段就补上（已存在的表不会报错）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(person)")}
        for col, ddl in [
            ("contact",       "ALTER TABLE person ADD COLUMN contact TEXT DEFAULT ''"),
            ("consent_at",    "ALTER TABLE person ADD COLUMN consent_at TEXT DEFAULT ''"),
            ("consent_scope", "ALTER TABLE person ADD COLUMN consent_scope TEXT DEFAULT ''"),
        ]:
            if col not in cols:
                conn.execute(ddl)


def load_db():
    out = {}
    with _lock, _conn() as conn:
        for name, blob in conn.execute("SELECT name, feature FROM person"):
            out[name] = np.frombuffer(blob, dtype=np.float32).copy()
    return out


def save_db(db):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM person")
        conn.executemany(
            "INSERT INTO person(name, feature) VALUES(?, ?)",
            [(n, np.asarray(f, dtype=np.float32).tobytes()) for n, f in db.items()]
        )


def add_person(name, feature, contact="", consent_remark=""):
    """新增/更新人员。contact 只有对方授权保存时才传。"""
    import time
    blob = np.asarray(feature, dtype=np.float32).tobytes()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with _lock, _conn() as conn:
        conn.execute("""
            INSERT INTO person(name, feature, contact, consent_at, consent_scope)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                feature       = excluded.feature,
                contact       = excluded.contact,
                consent_at    = excluded.consent_at,
                consent_scope = excluded.consent_scope,
                updated_at    = datetime('now','localtime')
        """, (name, blob, contact or "", now, "enroll,query,contact"))
    print(f"[OK] 已保存 {name}，当前库内共 {count()} 人")


def get_person(name):
    """按姓名取完整档案（含联系方式）"""
    with _lock, _conn() as conn:
        r = conn.execute(
            "SELECT id, name, contact, created_at, updated_at FROM person WHERE name = ?",
            (name,)
        ).fetchone()
        if not r:
            return None
        return {"id": r[0], "name": r[1], "contact": r[2],
                "created_at": r[3], "updated_at": r[4]}


def delete_person(name):
    with _lock, _conn() as conn:
        cur = conn.execute("DELETE FROM person WHERE name = ?", (name,))
        return cur.rowcount > 0


def list_people():
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT id, name, contact, created_at, updated_at FROM person ORDER BY id"
        )
        return [
            {"id": r[0], "name": r[1], "contact": r[2] or "",
             "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]


def list_persons():
    return list_people()


def count():
    with _lock, _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM person").fetchone()[0]


init_db()
