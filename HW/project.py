"""발진성 피부 질환 감별 시스템 - Streamlit 프로토타입 뼈대
실행: streamlit run app.py
필요 패키지: streamlit, numpy, pillow, opencv-python
(모델 연결 전까지 분류 결과는 '임시값'입니다. predict_image() 만 교체하면 됩니다.)
"""
import numpy as np
import streamlit as st
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# 낮은 확신도 경고 기준 (회의에서 값 조정)
LOW_CONF_TOP = 0.50     # 1위 확률이 이보다 낮으면 경고
LOW_CONF_MARGIN = 0.10  # 1위와 2위 확률 차이가 이보다 작으면 경고

CLASSES = {
    "AD": "아토피 피부염",
    "CD": "접촉성 피부염",
    "EC": "습진",
    "SC": "옴",
    "SD": "지루성 피부염",
    "TC": "체부백선",
}

# ------------------------------------------------------------------
# 1. 전처리 모듈 (3-1)
# ------------------------------------------------------------------
def preprocess(img_rgb: np.ndarray, use_clahe=True, gamma=1.0, denoise=True) -> np.ndarray:
    out = img_rgb.copy()
    if not HAS_CV2:
        return out
    if gamma != 1.0:  # Intensity transformation
        table = ((np.arange(256) / 255.0) ** (1 / gamma) * 255).astype("uint8")
        out = cv2.LUT(out, table)
    if use_clahe:  # Histogram processing (밝기 채널만)
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if denoise:  # Median filtering
        out = cv2.medianBlur(out, 5)
    return out


# ------------------------------------------------------------------
# 2. 병변 분할 모듈 (3-2)  -- 임계값 방식 (나중에 딥러닝으로 교체 가능)
# ------------------------------------------------------------------
def segment_lesion(img_rgb: np.ndarray, normal_rgb: np.ndarray | None = None):
    """반환: (mask(0/255), 지표 dict). 정상 피부 사진이 있으면 색 차이 기반."""
    if not HAS_CV2:
        return None, {}
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype("float32")
    if normal_rgb is not None:
        ref = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2LAB).astype("float32")
        ref_mean = ref.reshape(-1, 3).mean(axis=0)
    else:
        ref_mean = np.median(lab.reshape(-1, 3), axis=0)  # 이미지 중앙값을 '정상 피부'로 가정
    # a*(붉은기) 방향 차이를 병변 점수로 사용
    score = np.clip(lab[:, :, 1] - ref_mean[1], 0, None)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = int((mask > 0).sum())
    metrics = {
        "병변 면적 비율(%)": round(100 * area / mask.size, 1),
        "병변 개수": len(contours),
    }
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        a = cv2.contourArea(c)
        metrics["원형도(최대 병변)"] = round(4 * np.pi * a / (peri ** 2 + 1e-6), 2)
    return mask, metrics


def overlay(img_rgb, mask):
    if mask is None:
        return img_rgb
    out = img_rgb.copy()
    out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array([0, 255, 0])).astype("uint8")
    return out


# ------------------------------------------------------------------
# 3. AI 분류 모듈 (3-5)  -- TODO: 실제 모델로 교체
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    # TODO: 학습한 CNN 로드 (예: torch.load("model.pt")) / 외부 모델 불러오기
    #
    # ※ 클래스 가중치(class weight)는 이 파일(app.py)이 아니라 '학습 스크립트/노트북'에 넣습니다.
    #   웹은 이미 학습된 모델 파일만 불러와 예측하기 때문입니다.
    #   학습 코드에서 loss를 정의하는 곳에 아래처럼 넣으세요:
    #
    #   counts = np.array([70, 477, 466, 343, 79, 275])   # AD, CD, EC, SC, SD, TC 순서
    #   weights = counts.sum() / (len(counts) * counts)    # 적은 클래스일수록 큰 가중치
    #   criterion = torch.nn.CrossEntropyLoss(
    #       weight=torch.tensor(weights, dtype=torch.float32))
    #
    #   (클래스 순서는 CLASSES 딕셔너리 순서와 반드시 같아야 합니다)
    return None


def predict_image(img_rgb: np.ndarray) -> dict:
    """반환: {클래스코드: 확률}. 지금은 임시 난수(데모용)."""
    model = load_model()
    if model is None:
        rng = np.random.default_rng(int(img_rgb.mean()))
        p = rng.dirichlet(np.ones(len(CLASSES)))
        return dict(zip(CLASSES.keys(), p))
    # TODO: 전처리 -> 텐서 변환 -> model(x) -> softmax
    raise NotImplementedError


# ------------------------------------------------------------------
# 4. 문진 보정 모듈 (3-4)  -- 규칙 기반 가중치 (논문 근거로 값 채우기)
# ------------------------------------------------------------------
def adjust_with_symptoms(probs: dict, ans: dict) -> tuple[dict, list[str]]:
    w = {k: 1.0 for k in probs}
    reasons = []
    # 아래 가중치는 '자리표시자'입니다. 체크리스트 논문 스터디 후 근거와 함께 수정하세요.
    if ans["night_itch"] and ans["family_itch"]:
        w["SC"] *= 1.5
        reasons.append("밤에 심한 가려움 + 가족/동거인 유사 증상 → 옴 가능성 가중")
    if ans["new_contact"]:
        w["CD"] *= 1.4
        reasons.append("특정 물질 접촉 후 발생 → 접촉성 피부염 가능성 가중")
    if ans["pet_contact"]:
        w["TC"] *= 1.3
        reasons.append("반려동물 접촉 → 체부백선 가능성 가중")
    if ans["spreading"] and ans["ring_shape"]:
        w["TC"] *= 1.4
        reasons.append("가장자리가 퍼지는 고리 모양 → 체부백선 가능성 가중")
    if ans["chronic_dry"]:
        w["AD"] *= 1.3
        reasons.append("만성 건조/재발성 → 아토피 피부염 가능성 가중")
    adj = {k: probs[k] * w[k] for k in probs}
    s = sum(adj.values())
    return {k: v / s for k, v in adj.items()}, reasons


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="발진성 피부 질환 감별", page_icon="🩺", layout="wide")
st.title("🩺 발진성 피부 질환 감별 시스템")
st.caption("병변 추출 및 AI 기반 질환 분류 · 진단/처방이 아닌 감별 및 추가 확인을 위한 보조 도구입니다.")

if not HAS_CV2:
    st.warning("opencv-python 이 설치되지 않아 전처리/분할이 비활성화됩니다. `pip install opencv-python`")

with st.sidebar:
    st.header("전처리 옵션")
    use_clahe = st.checkbox("히스토그램 평활화(CLAHE)", True)
    gamma = st.slider("감마(밝기) 보정", 0.5, 2.0, 1.0, 0.1)
    denoise = st.checkbox("Median 노이즈 제거", True)
    st.divider()
    st.markdown("**촬영 가이드**\n- 자연광, 그림자 없이\n- 같은 조건에서 정상 피부도 촬영\n- 병변이 화면 중앙, 10~20cm 거리")

tab1, tab2, tab3 = st.tabs(["① 사진 업로드", "② 증상·생활 문진", "③ 분석 결과"])

with tab1:
    c1, c2 = st.columns(2)
    lesion_file = c1.file_uploader("병변 피부 사진 (필수)", type=["jpg", "jpeg", "png"], key="lesion")
    normal_file = c2.file_uploader("같은 조건의 정상 피부 사진 (선택)", type=["jpg", "jpeg", "png"], key="normal")
    if lesion_file:
        c1.image(lesion_file, caption="병변 사진", use_container_width=True)
    if normal_file:
        c2.image(normal_file, caption="정상 피부 사진", use_container_width=True)

with tab2:
    st.subheader("증상 정도 (단순 상/중/하가 아닌 구체적 기준)")
    itch = st.radio(
        "가려움 정도",
        ["없음", "가끔 가렵지만 신경 쓰이지 않음", "긁고 싶지만 참을 수 있음",
         "자주 긁게 되어 일상에 지장", "참기 어려워 피가 나도록 긁음"],
        index=0,
    )
    pain = st.radio("통증/화끈거림", ["없음", "약간 따가움", "화끈거려 신경 쓰임", "통증으로 일상에 지장"])
    c1, c2 = st.columns(2)
    with c1:
        night_itch = st.checkbox("밤에 증상이 더 심해진다")
        spreading = st.checkbox("점점 퍼지고 있다")
        ring_shape = st.checkbox("가장자리가 붉고 가운데가 옅은 고리 모양이다")
        chronic_dry = st.checkbox("피부가 건조하고 재발을 반복해 왔다")
    with c2:
        new_contact = st.checkbox("새 화장품/세제/금속 등과 접촉 후 시작됐다")
        family_itch = st.checkbox("가족·동거인도 비슷한 증상이 있다")
        pet_contact = st.checkbox("반려동물과 접촉했다")
        scaling = st.checkbox("각질/비늘이 일어난다")
    onset = st.selectbox("증상 시작 시기", ["3일 이내", "1~2주", "1개월 이내", "1개월 이상"])
    body_part = st.multiselect("발생 부위", ["얼굴/두피", "목", "몸통", "팔", "손", "다리", "발", "사타구니/겨드랑이"])
    st.subheader("의료 이력")
    hospital = st.text_area("이 증상으로 병원에 간 적이 있다면 진단/처방 내용", height=70)
    disease = st.text_input("지병 (예: 당뇨, 천식)")
    meds = st.text_input("복용 중인 약물 (스테로이드 연고/항생제 포함)")

with tab3:
    if st.button("분석하기", type="primary", disabled=lesion_file is None):
        img = np.array(Image.open(lesion_file).convert("RGB"))
        normal = np.array(Image.open(normal_file).convert("RGB")) if normal_file else None

        proc = preprocess(img, use_clahe, gamma, denoise)
        normal_proc = preprocess(normal, use_clahe, gamma, denoise) if normal is not None else None
        mask, metrics = segment_lesion(proc, normal_proc)

        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="원본", use_container_width=True)
        c2.image(proc, caption="전처리 후", use_container_width=True)
        c3.image(overlay(proc, mask), caption="병변 추출(초록)", use_container_width=True)
        if metrics:
            st.subheader("병변 정량 지표")
            st.table(metrics)

        probs = predict_image(proc)
        answers = dict(night_itch=night_itch, family_itch=family_itch, new_contact=new_contact,
                       pet_contact=pet_contact, spreading=spreading, ring_shape=ring_shape,
                       chronic_dry=chronic_dry)
        adj, reasons = adjust_with_symptoms(probs, answers)

        st.subheader("감별 후보")
        ranked = sorted(adj.items(), key=lambda x: -x[1])
        for code, p in ranked[:3]:
            st.write(f"**{CLASSES[code]}**  —  {p*100:.1f}%")
            st.progress(float(p))
        top = ranked[0]
        margin = ranked[0][1] - ranked[1][1]
        if top[1] < LOW_CONF_TOP or margin < LOW_CONF_MARGIN:
            st.error("⚠️ 결과의 확신도가 낮습니다. 상위 후보 간 차이가 작거나 최고 확률이 낮아 "
                     "이 결과만으로 판단하기 어렵습니다. 의료진 진료를 권장합니다.")
        st.info(f"가장 가능성이 높은 후보는 **{CLASSES[top[0]]}** 입니다. "
                f"{CLASSES[ranked[1][0]]}와(과) 감별이 필요할 수 있습니다.")

        with st.expander("판단 근거 / 추가 확인 정보", expanded=True):
            st.markdown("**문진 기반 보정 근거**")
            for r in reasons or ["문진에서 특이 가중 요인 없음"]:
                st.write("- " + r)
            st.markdown("**영상 vs 문진 확률 비교**")
            st.bar_chart({"영상만": [probs[k] for k in CLASSES],
                          "영상+문진": [adj[k] for k in CLASSES]})
            st.caption("현재 영상 확률은 모델 연결 전 임시값입니다.")
        st.warning("본 결과는 진단이 아니며, 증상이 지속·악화되면 반드시 의료진의 진료를 받으세요.")
    elif lesion_file is None:
        st.write("먼저 ① 탭에서 병변 사진을 업로드해 주세요.")
