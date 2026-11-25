# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
from skimage import color, filters, morphology, segmentation, measure

MAX_SIDE = 800


def fit(img, max_side=MAX_SIDE):
    """긴 변을 max_side로 줄여 속도/파라미터 스케일을 통일"""
    img = img.convert("RGB")
    s = max_side / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s))) if s < 1 else img


def feature_map(rgb, l_weight=0.5, blur=2):
    """배경색과의 Lab 거리 (0~1). 배경색은 이미지 테두리의 중앙값으로 추정.
    → 배경/객체의 색·밝기가 어떻든 '배경과 다른 정도'로 통일되어 범용적으로 동작"""
    lab = color.rgb2lab(rgb)
    lab = np.stack([filters.gaussian(lab[..., i], blur) for i in range(3)], -1)
    h, w = lab.shape[:2]
    m = max(4, min(h, w) // 25)
    border = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
                             lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    d = lab - np.median(border, 0)
    dist = np.sqrt((l_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    return dist / (dist.max() + 1e-9)


def segment(rgb, thr=None, min_area=0.0005, open_r=2, drop_border=False):
    f = feature_map(rgb)
    t = filters.threshold_otsu(f) if thr is None else thr      # thr=None → 자동(Otsu)
    mask = morphology.opening(f > t, morphology.disk(open_r))   # 잡음 제거
    mask = nd.binary_fill_holes(mask)                           # 구멍 메우기
    mask = morphology.remove_small_objects(mask, max_size=int(min_area * mask.size))
    lab = measure.label(mask)
    if drop_border:
        lab = segmentation.clear_border(lab)
    lab, _, _ = segmentation.relabel_sequential(lab)
    return f, t, lab


def analyze(lab):
    """덩어리별 면적/중심 + 추정 개수. 붙어 있는 덩어리는 '중앙값 면적'의 배수로 개수 추정"""
    props = measure.regionprops(lab)
    if not props:
        return pd.DataFrame(columns=["id", "x", "y", "area", "count"])
    areas = np.array([p.area for p in props])
    ref = np.median(areas)
    cnt = np.maximum(1, np.floor(areas / ref + 0.4).astype(int))
    return pd.DataFrame({"id": [p.label for p in props],
                         "x": [int(p.centroid[1]) for p in props],
                         "y": [int(p.centroid[0]) for p in props],
                         "area": areas.astype(int), "count": cnt})


def overlay(img, lab, df):
    """원본 위에 윤곽선 + 번호(붙은 덩어리는 xN) 표시"""
    arr = np.asarray(img).copy()
    arr[segmentation.find_boundaries(lab, mode="outer")] = (255, 40, 40)
    out = Image.fromarray(arr)
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=max(12, img.width // 45))
    for r in df.itertuples():
        txt = str(r.id) if r.count == 1 else f"{r.id}(x{r.count})"
        d.text((r.x, r.y), txt, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0),
               font=font, anchor="mm")
    return out


# ---------------------------------------------------------------- UI
import io
import streamlit as st
from PIL import ImageOps


def process_image(upload):
    image = ImageOps.exif_transpose(Image.open(upload))     # 휴대폰 사진 회전 보정
    img = fit(image)
    rgb = np.asarray(img) / 255.0

    f, otsu, lab = segment(rgb, thr=thr, min_area=min_area / 100, open_r=open_r, drop_border=drop_border)
    df = analyze(lab)
    result = overlay(img, lab, df)

    col1.write("original")
    col1.image(img, use_container_width=True)
    col2.write("foreground mask")
    col2.image((lab > 0).astype(np.uint8) * 255, use_container_width=True)
    col3.write("counting result")
    col3.image(result, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total count", int(df["count"].sum()))
    m2.metric("Blobs (연결 덩어리)", len(df))
    m3.metric("Median area (px)", int(df["area"].median()) if len(df) else 0)
    m4.metric("Threshold", f"{otsu if thr is None else thr:.2f}" + (" (Otsu)" if thr is None else ""))

    with st.expander("Object table", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
    buf = io.BytesIO(); result.save(buf, "PNG")
    d1, d2 = st.columns(2)
    d1.download_button("Download result (PNG)", buf.getvalue(), "result.png", "image/png")
    d2.download_button("Download table (CSV)", df.to_csv(index=False).encode(), "objects.csv", "text/csv")


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="DIP")
    st.write("## Object Counting System")
    st.write(":cat:")
    st.sidebar.write("## Settings")
    my_upload = st.sidebar.file_uploader("Upload", type=["png", "jpg", "jpeg"])
    auto = st.sidebar.toggle("Auto threshold (Otsu)", value=True)
    thr = None if auto else st.sidebar.slider("Threshold", 0.05, 0.95, 0.30, 0.01)
    min_area = st.sidebar.slider("Min object size (% of image)", 0.01, 0.5, 0.05, 0.01)
    open_r = st.sidebar.slider("Noise removal (px)", 0, 6, 2)
    drop_border = st.sidebar.checkbox("Ignore objects touching the border", value=False)

    col1, col2, col3 = st.columns(3)
    process_image(my_upload if my_upload is not None else "./gr.jpeg")
