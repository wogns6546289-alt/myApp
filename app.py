from pathlib import Path
from typing import List, Tuple
import io

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "best.pt"

SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]


# ------------------------------------------------------------
# Streamlit 페이지 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="YOLO 비전 검사",
    page_icon="🔍",
    layout="wide"
)


# ------------------------------------------------------------
# 모델 로드
# Streamlit이 화면을 다시 실행할 때마다 모델을 재로딩하지 않도록 캐시
# ------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {path.resolve()}"
        )

    return YOLO(str(path))


# ------------------------------------------------------------
# 탐지 결과를 표 형태로 변환
# ------------------------------------------------------------
def make_detection_rows(result, model: YOLO) -> List[dict]:
    rows = []

    if result.boxes is None:
        return rows

    for index, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        xyxy = box.xyxy.detach().cpu().numpy().ravel()

        if len(xyxy) != 4:
            continue

        x1, y1, x2, y2 = map(float, xyxy)

        rows.append(
            {
                "번호": index + 1,
                "클래스 ID": class_id,
                "클래스명": model.names[class_id],
                "신뢰도": round(confidence, 4),
                "X1": round(x1, 1),
                "Y1": round(y1, 1),
                "X2": round(x2, 1),
                "Y2": round(y2, 1),
                "폭": round(x2 - x1, 1),
                "높이": round(y2 - y1, 1),
            }
        )

    return rows


# ------------------------------------------------------------
# PIL 이미지를 PNG 바이트로 변환
# 다운로드 버튼에서 사용
# ------------------------------------------------------------
def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ------------------------------------------------------------
# 단일 이미지 예측
# ------------------------------------------------------------
def predict_image(
    model: YOLO,
    image: Image.Image,
    confidence: float,
    image_size: int,
    line_width: int
) -> Tuple[Image.Image, object, List[dict]]:

    # YOLO 입력용 RGB numpy 배열
    image_rgb = image.convert("RGB")
    image_array = np.array(image_rgb)

    results = model.predict(
        source=image_array,
        conf=confidence,
        imgsz=image_size,
        verbose=False
    )

    if not results:
        raise RuntimeError("YOLO 예측 결과가 반환되지 않았습니다.")

    result = results[0]

    # result.plot() 반환 이미지는 BGR 순서
    annotated_bgr = result.plot(
        boxes=True,
        labels=True,
        conf=True,
        line_width=line_width
    )

    # BGR → RGB 변환
    annotated_rgb = annotated_bgr[:, :, ::-1]

    annotated_image = Image.fromarray(annotated_rgb)
    detection_rows = make_detection_rows(result, model)

    return annotated_image, result, detection_rows


# ------------------------------------------------------------
# 제목
# ------------------------------------------------------------
st.title("YOLO 비전 검사 서비스")

st.write(
    "이미지를 업로드하면 학습된 YOLO 모델로 객체를 탐지하고 "
    "원본 이미지와 탐지 결과를 비교합니다."
)


# ------------------------------------------------------------
# 모델 상태 표시
# ------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        f"모델 파일을 찾을 수 없습니다.\n\n"
        f"`{MODEL_PATH}` 위치에 `best.pt`를 넣어주세요."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.exception(error)
    st.stop()


# ------------------------------------------------------------
# 사이드바 설정
# ------------------------------------------------------------
with st.sidebar:
    st.header("검사 설정")

    confidence = st.slider(
        "최소 신뢰도",
        min_value=0.01,
        max_value=1.00,
        value=0.05,
        step=0.01,
        help="이 값보다 낮은 신뢰도의 탐지는 표시하지 않습니다."
    )

    image_size = st.select_slider(
        "추론 이미지 크기",
        options=[320, 416, 512, 640, 800, 960, 1280],
        value=320,
        help="학습 시 imgsz=320을 사용했다면 우선 320으로 검사하세요."
    )

    line_width = st.slider(
        "바운딩 박스 선 굵기",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    st.divider()

    st.write("모델 경로")

    st.code(
        str(MODEL_PATH.relative_to(APP_DIR)),
        language=None
    )

    st.success("모델 로드 완료")


# ------------------------------------------------------------
# 이미지 업로드
# ------------------------------------------------------------
uploaded_files = st.file_uploader(
    "검사할 이미지를 선택하세요.",
    type=SUPPORTED_FILE_TYPES,
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("위 영역에서 검사할 이미지를 업로드하세요.")
    st.stop()


# ------------------------------------------------------------
# 예측 실행
# ------------------------------------------------------------
if st.button(
    "검사 시작",
    type="primary",
    use_container_width=True
):
    st.session_state["run_prediction"] = True


if not st.session_state.get("run_prediction", False):
    st.stop()


progress_bar = st.progress(0)
status_text = st.empty()

total_detected_count = 0
summary_rows = []

for file_index, uploaded_file in enumerate(uploaded_files):
    status_text.write(
        f"검사 중: {uploaded_file.name} "
        f"({file_index + 1}/{len(uploaded_files)})"
    )

    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        annotated_image, result, detection_rows = predict_image(
            model=model,
            image=original_image,
            confidence=confidence,
            image_size=image_size,
            line_width=line_width
        )

        detected_count = len(detection_rows)
        total_detected_count += detected_count

        summary_rows.append(
            {
                "파일명": uploaded_file.name,
                "탐지 개수": detected_count,
                "판정": "탐지됨" if detected_count > 0 else "탐지 없음"
            }
        )

        st.divider()
        st.subheader(
            f"{file_index + 1}. {uploaded_file.name}"
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("탐지 개수", detected_count)

        with metric_col2:
            st.metric("이미지 폭", original_image.width)

        with metric_col3:
            st.metric("이미지 높이", original_image.height)

        original_col, prediction_col = st.columns(2)

        with original_col:
            st.markdown("#### 원본 이미지")
            st.image(
                original_image,
                use_container_width=True
            )

        with prediction_col:
            st.markdown("#### 예측 결과")
            st.image(
                annotated_image,
                use_container_width=True
            )

        if detection_rows:
            st.markdown("#### 탐지 상세 정보")

            detection_dataframe = pd.DataFrame(detection_rows)

            st.dataframe(
                detection_dataframe,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(
                f"신뢰도 {confidence:.2f} 이상으로 탐지된 객체가 없습니다."
            )

        output_filename = (
            f"{Path(uploaded_file.name).stem}_prediction.png"
        )

        st.download_button(
            label="예측 결과 이미지 다운로드",
            data=image_to_png_bytes(annotated_image),
            file_name=output_filename,
            mime="image/png",
            key=f"download_{file_index}"
        )

    except Exception as error:
        st.error(
            f"`{uploaded_file.name}` 검사 중 오류가 발생했습니다."
        )
        st.exception(error)

        summary_rows.append(
            {
                "파일명": uploaded_file.name,
                "탐지 개수": "-",
                "판정": "검사 오류"
            }
        )

    progress_bar.progress(
        (file_index + 1) / len(uploaded_files)
    )


status_text.success("모든 이미지 검사가 완료되었습니다.")


# ------------------------------------------------------------
# 전체 검사 요약
# ------------------------------------------------------------
st.divider()
st.header("전체 검사 요약")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.metric("검사 이미지 수", len(uploaded_files))

with summary_col2:
    st.metric("전체 탐지 개수", total_detected_count)

if summary_rows:
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True
    )
