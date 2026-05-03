"""Face identifier — take a webcam photo, recognize known faces, label unknowns.

Uses InsightFace's pretrained `buffalo_s` pack (SCRFD detector + ArcFace MBF
recognition) for accurate 512-d face embeddings. Each known person is a folder
under `known_faces/<name>/` containing `.jpg` crops and `.npy` embeddings.

Run it either way:
  python app.py            (auto-launches Streamlit and opens your browser)
  streamlit run app.py
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path


# --- Self-launch when invoked as `python app.py` -----------------------------
def _under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _under_streamlit():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__), *sys.argv[1:]]
    sys.exit(stcli.main())
# -----------------------------------------------------------------------------

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from insightface.app import FaceAnalysis

KNOWN_DIR = Path(__file__).parent / "known_faces"
KNOWN_DIR.mkdir(exist_ok=True)
THRESHOLD = 0.42  # cosine similarity required to call it a match


@st.cache_resource(show_spinner="Loading face recognition model…")
def get_model() -> FaceAnalysis:
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    app = FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(480, 480))
    return app


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\- ]", "", name).strip().replace(" ", "_")


def list_people() -> list[str]:
    return sorted(p.name for p in KNOWN_DIR.iterdir() if p.is_dir())


def load_db() -> dict[str, np.ndarray]:
    db: dict[str, np.ndarray] = {}
    for person in KNOWN_DIR.iterdir():
        if not person.is_dir():
            continue
        embs = [np.load(f) for f in person.glob("*.npy")]
        if not embs:
            continue
        avg = np.mean(np.stack(embs), axis=0)
        avg /= (np.linalg.norm(avg) + 1e-8)
        db[person.name] = avg.astype(np.float32)
    return db


def save_face(name: str, crop_bgr: np.ndarray, emb: np.ndarray) -> None:
    folder = KNOWN_DIR / safe_name(name)
    folder.mkdir(exist_ok=True)
    fid = uuid.uuid4().hex[:10]
    cv2.imwrite(str(folder / f"{fid}.jpg"), crop_bgr)
    np.save(str(folder / f"{fid}.npy"), emb.astype(np.float32))


def identify_faces(faces, db: dict[str, np.ndarray]):
    """Return list of (face, matched_name_or_None, similarity)."""
    if not db:
        return [(f, None, 0.0) for f in faces]
    names = list(db.keys())
    matrix = np.stack([db[n] for n in names])
    out = []
    for f in faces:
        sims = matrix @ f.normed_embedding
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        out.append((f, names[idx] if sim >= THRESHOLD else None, sim))
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Face Identifier", layout="wide")
st.title("Face Identifier")
st.caption("Take a photo with your webcam. Known faces are labeled in green; unknown faces appear on the right to be named.")

model = get_model()
db = load_db()

left, right = st.columns([3, 2], gap="large")

with left:
    photo = st.camera_input("Webcam")

unknowns: list[tuple[np.ndarray, np.ndarray]] = []  # (crop_bgr, embedding)

if photo is not None:
    img_rgb = np.array(Image.open(photo).convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    faces = model.get(img_bgr)
    results = identify_faces(faces, db)

    annotated = img_bgr.copy()
    h, w = annotated.shape[:2]
    for f, name, sim in results:
        x1, y1, x2, y2 = (max(0, int(v)) for v in f.bbox)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = (0, 200, 0) if name else (0, 0, 220)
        label = f"{name}  {sim:.2f}" if name else f"Unknown  {sim:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if name is None and x2 > x1 and y2 > y1:
            unknowns.append((img_bgr[y1:y2, x1:x2].copy(), f.normed_embedding.copy()))

    with left:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"{len(results)} face(s) detected")

with right:
    st.subheader("Identify Unknown Faces")
    people = list_people()

    if photo is None:
        st.info("Take a photo on the left to start.")
    elif not unknowns:
        st.success("All faces in the photo are recognized — nothing to label.")
    else:
        st.caption(f"{len(unknowns)} unknown face(s) in this photo.")

    for i, (crop, emb) in enumerate(unknowns):
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width="stretch")
            with c2:
                key = f"u{i}"
                name = st.text_input("Name / folder", key=f"name_{key}", placeholder="e.g. Alex Kim")
                conflict_key = f"conflict_{key}"

                if st.button("Save", key=f"save_{key}", type="primary"):
                    clean = safe_name(name or "")
                    if not clean:
                        st.warning("Please enter a name.")
                    elif clean in people:
                        st.session_state[conflict_key] = clean
                        st.rerun()
                    else:
                        save_face(clean, crop, emb)
                        st.toast(f"Saved to new folder '{clean}'")
                        st.rerun()

                if st.session_state.get(conflict_key):
                    existing = st.session_state[conflict_key]
                    st.warning(f"A folder named **{existing}** already exists. "
                               "Is this the same person?")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("Yes — add to this folder", key=f"same_{key}"):
                            save_face(existing, crop, emb)
                            st.session_state.pop(conflict_key, None)
                            st.toast(f"Added face to '{existing}'")
                            st.rerun()
                    with cc2:
                        if st.button("No — pick a new name", key=f"diff_{key}"):
                            st.session_state.pop(conflict_key, None)
                            st.rerun()

    st.divider()
    st.caption("**Known people:** " + (", ".join(f"{p} ({len(list((KNOWN_DIR/p).glob('*.npy')))})" for p in people) if people else "(none yet)"))
