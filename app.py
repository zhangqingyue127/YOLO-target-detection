"""Streamlit Web 演示界面：上传图片/视频并展示检测结果"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

WEIGHT_PATH = "runs/detect/yolo_detection/weights/best.pt"

st.set_page_config(page_title="YOLO 目标检测", layout="wide")
st.title("YOLO 目标检测演示系统")


@st.cache_resource
def load_model():
    return YOLO(WEIGHT_PATH)


if not Path(WEIGHT_PATH).exists():
    st.error(f"未找到模型权重 `{WEIGHT_PATH}`，请先完成训练。")
    st.stop()

model = load_model()

tab1, tab2 = st.tabs(["图片检测", "视频检测"])

with tab1:
    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    conf = st.slider("置信度阈值", 0.1, 1.0, 0.3, 0.05)
    if uploaded:
        image = Image.open(uploaded)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = model(image_bgr, conf=conf)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="检测结果", use_container_width=True)

with tab2:
    uploaded = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    conf = st.slider("置信度阈值", 0.1, 1.0, 0.3, 0.05, key="vid_conf")
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            results = model(frame, conf=conf, verbose=False)
            annotated = results[0].plot()
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          channels="RGB")
        cap.release()

st.markdown("---")
st.caption(f"模型权重: {WEIGHT_PATH} | 类别数: {len(model.names)}")
