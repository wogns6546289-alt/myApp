from pathlib import Path
from typing import List, Tuple, Optional
import io
import json

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

from google import genai
from google.genai import types


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "best.pt"

SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

# secrets.toml에 GEMINI_MODEL이 없을 때 사용할 기본 모델
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


# ------------------------------------------------------------
# Streamlit 페이지 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Screw 비전 검사",
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
# Streamlit Secrets에서 Gemini API 키 읽기
# ------------------------------------------------------------
def get_gemini_api_key() -> Optional[str]:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")

        if api_key is None:
            return None

        api_key = str(api_key).strip()

        if not api_key:
            return None

        return api_key

    except Exception:
        return None


# ------------------------------------------------------------
# Streamlit Secrets에서 Gemini 모델명 읽기
# ------------------------------------------------------------
def get_gemini_model_name() -> str:
    try:
        model_name = st.secrets.get(
            "GEMINI_MODEL",
            DEFAULT_GEMINI_MODEL
        )

        model_name = str(model_name).strip()

        if not model_name:
            return DEFAULT_GEMINI_MODEL

        return model_name

    except Exception:
        return DEFAULT_GEMINI_MODEL


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

        # model.names는 dict 또는 list 형태일 수 있으므로 방어 처리
        try:
            class_name = str(model.names[class_id])
        except Exception:
            class_name = f"class_{class_id}"

        rows.append(
            {
                "번호": index + 1,
                "클래스 ID": class_id,
                "클래스명": class_name,
                "신뢰도": round(confidence, 4),
                "X1": round(x1, 1),
                "Y1": round(y1, 1),
                "X2": round(x2, 1),
                "Y2": round(y2, 1),
                "폭": round(max(0.0, x2 - x1), 1),
                "높이": round(max(0.0, y2 - y1), 1),
            }
        )

    return rows


# ------------------------------------------------------------
# PIL 이미지를 PNG 바이트로 변환
# 다운로드 버튼 및 Gemini 이미지 입력에 사용
# ------------------------------------------------------------
def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="PNG",
        optimize=True
    )

    return buffer.getvalue()


# ------------------------------------------------------------
# Gemini 전송용 이미지 크기 축소
#
# 원본 해상도가 지나치게 크면 전송 시간과 API 처리 비용이 증가할 수
# 있으므로 최대 길이를 기준으로 축소한다.
# 원본 Streamlit 표시 이미지와 YOLO 결과 이미지는 변경하지 않는다.
# ------------------------------------------------------------
def resize_image_for_gemini(
    image: Image.Image,
    max_side: int = 1280
) -> Image.Image:

    image_rgb = image.convert("RGB")

    width, height = image_rgb.size

    if width <= 0 or height <= 0:
        raise ValueError("이미지 폭 또는 높이가 올바르지 않습니다.")

    longest_side = max(width, height)

    if longest_side <= max_side:
        return image_rgb.copy()

    scale = max_side / float(longest_side)

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    return image_rgb.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS
    )


# ------------------------------------------------------------
# 탐지 결과 기본 통계 생성
# ------------------------------------------------------------
def make_detection_statistics(
    detection_rows: List[dict]
) -> dict:

    if not detection_rows:
        return {
            "detected_count": 0,
            "class_counts": {},
            "average_confidence": None,
            "minimum_confidence": None,
            "maximum_confidence": None
        }

    confidence_values = []
    class_counts = {}

    for row in detection_rows:
        class_name = str(row.get("클래스명", "알 수 없음"))
        confidence = float(row.get("신뢰도", 0.0))

        class_counts[class_name] = (
            class_counts.get(class_name, 0) + 1
        )

        confidence_values.append(confidence)

    return {
        "detected_count": len(detection_rows),
        "class_counts": class_counts,
        "average_confidence": round(
            float(np.mean(confidence_values)),
            4
        ),
        "minimum_confidence": round(
            float(np.min(confidence_values)),
            4
        ),
        "maximum_confidence": round(
            float(np.max(confidence_values)),
            4
        )
    }


# ------------------------------------------------------------
# Gemini 입력용 프롬프트 생성
# ------------------------------------------------------------
def make_gemini_prompt(
    file_name: str,
    detection_rows: List[dict],
    image_width: int,
    image_height: int,
    confidence_threshold: float,
    image_size: int,
    inspection_context: str
) -> str:

    statistics = make_detection_statistics(detection_rows)

    detection_json = json.dumps(
        detection_rows,
        ensure_ascii=False,
        indent=2
    )

    class_count_json = json.dumps(
        statistics["class_counts"],
        ensure_ascii=False,
        indent=2
    )

    context_text = inspection_context.strip()

    if not context_text:
        context_text = (
            "나사 또는 산업용 부품 이미지에서 YOLO 객체 탐지 결과를 "
            "확인하는 비전 검사입니다."
        )

    prompt = f"""
당신은 산업용 비전 검사 결과를 분석하는 전문 엔지니어입니다.

아래에는 동일한 검사 대상의 이미지 2장이 순서대로 제공됩니다.

첫 번째 이미지:
- YOLO 검사에 사용한 원본 이미지

두 번째 이미지:
- YOLO가 탐지한 바운딩 박스, 클래스명, 신뢰도가 표시된 결과 이미지

검사 목적:
{context_text}

검사 파일:
- 파일명: {file_name}
- 원본 이미지 크기: {image_width} x {image_height}
- YOLO 추론 이미지 크기: {image_size}
- YOLO 최소 신뢰도 기준: {confidence_threshold:.2f}

YOLO 탐지 통계:
- 전체 탐지 개수: {statistics["detected_count"]}
- 클래스별 탐지 개수:
{class_count_json}
- 평균 신뢰도: {statistics["average_confidence"]}
- 최소 신뢰도: {statistics["minimum_confidence"]}
- 최대 신뢰도: {statistics["maximum_confidence"]}

YOLO 상세 탐지 데이터:
{detection_json}

다음 기준에 따라 한국어로 분석해 주세요.

1. 먼저 YOLO가 무엇을 탐지했는지 간단히 요약하세요.
2. 원본 이미지와 바운딩 박스 결과를 함께 확인하여 탐지가 시각적으로 타당한지 설명하세요.
3. 신뢰도가 낮거나 경계에 가까운 탐지가 있으면 명확히 표시하세요.
4. 잘못된 위치를 탐지했을 가능성, 중복 탐지 가능성, 객체 누락 가능성을 검토하세요.
5. 조명, 초점, 방향, 배경, 객체 크기 때문에 결과가 영향을 받았을 가능성이 있으면 설명하세요.
6. 탐지 결과가 없더라도 정상이라고 단정하지 말고 가능한 원인을 설명하세요.
7. 추가 학습이나 데이터 보강이 필요해 보인다면 구체적으로 어떤 이미지를 추가해야 하는지 제안하세요.
8. 사용자가 제공하지 않은 정상/불량 판정 기준을 임의로 만들어 내지 마세요.
9. 확실하지 않은 내용은 추정이라고 명확히 표시하세요.
10. AI 해석만으로 최종 품질 판정을 확정하지 마세요.
11. 각 항목을 너무 길게 설명하지말고, 핵심만 간단하게 해석하세요.

다음 형식으로 작성하세요.

### 검사 결과 요약
간단한 전체 요약

### YOLO 탐지 분석
탐지 개수, 클래스, 신뢰도 및 위치 분석

### 이미지 기반 검토
원본 이미지와 결과 이미지를 시각적으로 비교한 내용

### 주의할 부분
오탐, 미탐, 중복 탐지 또는 신뢰도가 낮은 부분

### 개선 제안
학습 데이터, 촬영 조건, confidence, imgsz 등에 대한 제안

### 결론
현재 결과를 어느 정도 신뢰할 수 있는지와 사람이 확인해야 할 사항

문장은 실무자가 이해하기 쉽게 작성하고, 지나치게 장황하게 작성하지 마세요.
"""

    return prompt.strip()


# ------------------------------------------------------------
# Gemini API를 이용한 결과 해석
# ------------------------------------------------------------
def analyze_with_gemini(
    api_key: str,
    gemini_model: str,
    file_name: str,
    original_image: Image.Image,
    annotated_image: Image.Image,
    detection_rows: List[dict],
    confidence_threshold: float,
    image_size: int,
    inspection_context: str
) -> str:

    if not api_key:
        raise ValueError("Gemini API 키가 설정되지 않았습니다.")

    if not gemini_model:
        raise ValueError("Gemini 모델명이 설정되지 않았습니다.")

    original_for_gemini = resize_image_for_gemini(
        original_image,
        max_side=1280
    )

    annotated_for_gemini = resize_image_for_gemini(
        annotated_image,
        max_side=1280
    )

    original_bytes = image_to_png_bytes(original_for_gemini)
    annotated_bytes = image_to_png_bytes(annotated_for_gemini)

    prompt = make_gemini_prompt(
        file_name=file_name,
        detection_rows=detection_rows,
        image_width=original_image.width,
        image_height=original_image.height,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        inspection_context=inspection_context
    )

    # context manager 종료 시 내부 HTTP 연결 리소스 해제
    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=gemini_model,
            contents=[
                types.Part.from_text(
                    text=prompt
                ),
                types.Part.from_bytes(
                    data=original_bytes,
                    mime_type="image/png"
                ),
                types.Part.from_bytes(
                    data=annotated_bytes,
                    mime_type="image/png"
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1500
            )
        )

    if response is None:
        raise RuntimeError(
            "Gemini API 응답이 반환되지 않았습니다."
        )

    response_text = getattr(response, "text", None)

    if response_text is None:
        raise RuntimeError(
            "Gemini API 응답에 텍스트 결과가 없습니다."
        )

    response_text = str(response_text).strip()

    if not response_text:
        raise RuntimeError(
            "Gemini API가 빈 해석 결과를 반환했습니다."
        )

    return response_text


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
        raise RuntimeError(
            "YOLO 예측 결과가 반환되지 않았습니다."
        )

    result = results[0]

    # result.plot() 반환 이미지는 BGR 순서
    annotated_bgr = result.plot(
        boxes=True,
        labels=True,
        conf=True,
        line_width=line_width
    )

    if annotated_bgr is None:
        raise RuntimeError(
            "YOLO 결과 이미지 생성에 실패했습니다."
        )

    # BGR → RGB 변환
    annotated_rgb = annotated_bgr[:, :, ::-1]

    # 음수 stride 배열 문제 방지를 위해 copy()
    annotated_image = Image.fromarray(
        annotated_rgb.copy()
    )

    detection_rows = make_detection_rows(
        result,
        model
    )

    return annotated_image, result, detection_rows


# ------------------------------------------------------------
# 제목
# ------------------------------------------------------------
st.title("YOLO 비전 검사 서비스")

st.write(
    "이미지를 업로드하면 학습된 YOLO 모델로 객체를 탐지하고, "
    "Google Gemini AI가 원본 이미지와 YOLO 결과를 함께 분석합니다."
)


# ------------------------------------------------------------
# API 키 및 모델명 확인
# ------------------------------------------------------------
gemini_api_key = get_gemini_api_key()
gemini_model_name = get_gemini_model_name()


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
        help=(
            "이 값보다 낮은 신뢰도의 YOLO 탐지는 "
            "표시하지 않습니다."
        )
    )

    image_size = st.select_slider(
        "추론 이미지 크기",
        options=[320, 416, 512, 640, 800, 960, 1280],
        value=320,
        help=(
            "학습 시 imgsz=320을 사용했다면 "
            "우선 320으로 검사하세요."
        )
    )

    line_width = st.slider(
        "바운딩 박스 선 굵기",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    st.divider()

    st.header("Gemini AI 설정")

    use_gemini = st.checkbox(
        "Gemini 결과 해석 사용",
        value=True,
        help=(
            "활성화하면 원본 이미지, YOLO 결과 이미지와 "
            "탐지 데이터를 Gemini API로 전송합니다."
        )
    )

    inspection_context = st.text_area(
        "검사 목적 및 정상/불량 기준",
        value=(
            "나사 이미지에서 결함 또는 이상 부위를 탐지하는 검사입니다. "
            "YOLO 바운딩 박스가 실제 이상 부위를 정확하게 가리키는지 "
            "중점적으로 확인해 주세요."
        ),
        height=140,
        help=(
            "Gemini가 임의로 판정하지 않도록 검사 대상과 "
            "정상/불량 기준을 구체적으로 작성하는 것이 좋습니다."
        )
    )

    st.write("Gemini 모델")

    st.code(
        gemini_model_name,
        language=None
    )

    if gemini_api_key:
        st.success("Gemini API 키 설정 완료")
    else:
        st.warning(
            "GEMINI_API_KEY가 설정되지 않았습니다."
        )

    st.divider()

    st.write("YOLO 모델 경로")

    st.code(
        str(MODEL_PATH.relative_to(APP_DIR)),
        language=None
    )

    st.success("YOLO 모델 로드 완료")


# ------------------------------------------------------------
# Gemini 설정 사전 확인
# ------------------------------------------------------------
if use_gemini and not gemini_api_key:
    st.warning(
        "Gemini 결과 해석이 활성화되어 있지만 API 키가 없습니다. "
        "YOLO 검사는 실행되지만 Gemini 해석은 생략됩니다."
    )


# ------------------------------------------------------------
# 이미지 업로드
# ------------------------------------------------------------
uploaded_files = st.file_uploader(
    "검사할 이미지를 선택하세요.",
    type=SUPPORTED_FILE_TYPES,
    accept_multiple_files=True
)

if not uploaded_files:
    st.info(
        "위 영역에서 검사할 이미지를 업로드하세요."
    )

    # 이전 검사 상태가 남아 있지 않도록 초기화
    st.session_state["run_prediction"] = False
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


if not st.session_state.get(
    "run_prediction",
    False
):
    st.stop()


progress_bar = st.progress(0)
status_text = st.empty()

total_detected_count = 0
gemini_success_count = 0
gemini_failure_count = 0

summary_rows = []


for file_index, uploaded_file in enumerate(uploaded_files):
    status_text.write(
        f"검사 중: {uploaded_file.name} "
        f"({file_index + 1}/{len(uploaded_files)})"
    )

    try:
        # 업로드 파일 포인터가 재사용될 가능성을 고려하여 처음으로 이동
        uploaded_file.seek(0)

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        annotated_image, result, detection_rows = predict_image(
            model=model,
            image=original_image,
            confidence=confidence,
            image_size=image_size,
            line_width=line_width
        )

        detected_count = len(detection_rows)
        total_detected_count += detected_count

        detection_statistics = make_detection_statistics(
            detection_rows
        )

        st.divider()

        st.subheader(
            f"{file_index + 1}. {uploaded_file.name}"
        )

        metric_col1, metric_col2, metric_col3, metric_col4 = (
            st.columns(4)
        )

        with metric_col1:
            st.metric(
                "탐지 개수",
                detected_count
            )

        with metric_col2:
            average_confidence = detection_statistics[
                "average_confidence"
            ]

            st.metric(
                "평균 신뢰도",
                (
                    f"{average_confidence:.4f}"
                    if average_confidence is not None
                    else "-"
                )
            )

        with metric_col3:
            st.metric(
                "이미지 폭",
                original_image.width
            )

        with metric_col4:
            st.metric(
                "이미지 높이",
                original_image.height
            )

        original_col, prediction_col = st.columns(2)

        with original_col:
            st.markdown("#### 원본 이미지")

            st.image(
                original_image,
                use_container_width=True
            )

        with prediction_col:
            st.markdown("#### YOLO 예측 결과")

            st.image(
                annotated_image,
                use_container_width=True
            )

        if detection_rows:
            st.markdown("#### 탐지 상세 정보")

            detection_dataframe = pd.DataFrame(
                detection_rows
            )

            st.dataframe(
                detection_dataframe,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning(
                f"신뢰도 {confidence:.2f} 이상으로 "
                f"탐지된 객체가 없습니다."
            )

        # ----------------------------------------------------
        # Gemini AI 결과 해석
        # ----------------------------------------------------
        gemini_analysis = None
        gemini_status = "사용 안 함"

        if use_gemini:
            st.markdown("#### Gemini AI 결과 해석")

            if not gemini_api_key:
                gemini_status = "API 키 없음"
                gemini_failure_count += 1

                st.warning(
                    "Gemini API 키가 없어 AI 해석을 "
                    "실행하지 않았습니다."
                )

            else:
                try:
                    with st.spinner(
                        "Gemini가 이미지와 YOLO 결과를 분석하고 있습니다..."
                    ):
                        gemini_analysis = analyze_with_gemini(
                            api_key=gemini_api_key,
                            gemini_model=gemini_model_name,
                            file_name=uploaded_file.name,
                            original_image=original_image,
                            annotated_image=annotated_image,
                            detection_rows=detection_rows,
                            confidence_threshold=confidence,
                            image_size=image_size,
                            inspection_context=inspection_context
                        )

                    gemini_status = "해석 완료"
                    gemini_success_count += 1

                    st.info(
                        "아래 내용은 Gemini가 원본 이미지와 "
                        "YOLO 결과를 함께 분석한 참고 의견입니다."
                    )

                    st.markdown(gemini_analysis)

                except Exception as gemini_error:
                    gemini_status = "해석 오류"
                    gemini_failure_count += 1

                    st.error(
                        "Gemini 결과 해석 중 오류가 발생했습니다. "
                        "YOLO 검사 결과에는 영향을 주지 않습니다."
                    )

                    st.exception(gemini_error)

        # ----------------------------------------------------
        # 전체 요약 데이터
        # ----------------------------------------------------
        summary_rows.append(
            {
                "파일명": uploaded_file.name,
                "탐지 개수": detected_count,
                "평균 신뢰도": (
                    detection_statistics["average_confidence"]
                    if detection_statistics[
                        "average_confidence"
                    ] is not None
                    else "-"
                ),
                "최소 신뢰도": (
                    detection_statistics["minimum_confidence"]
                    if detection_statistics[
                        "minimum_confidence"
                    ] is not None
                    else "-"
                ),
                "최대 신뢰도": (
                    detection_statistics["maximum_confidence"]
                    if detection_statistics[
                        "maximum_confidence"
                    ] is not None
                    else "-"
                ),
                "YOLO 결과": (
                    "탐지됨"
                    if detected_count > 0
                    else "탐지 없음"
                ),
                "Gemini 상태": gemini_status
            }
        )

        output_filename = (
            f"{Path(uploaded_file.name).stem}_prediction.png"
        )

        st.download_button(
            label="예측 결과 이미지 다운로드",
            data=image_to_png_bytes(annotated_image),
            file_name=output_filename,
            mime="image/png",
            key=f"download_image_{file_index}"
        )

        if gemini_analysis:
            analysis_filename = (
                f"{Path(uploaded_file.name).stem}"
                f"_gemini_analysis.txt"
            )

            st.download_button(
                label="Gemini 해석 결과 다운로드",
                data=gemini_analysis.encode("utf-8"),
                file_name=analysis_filename,
                mime="text/plain",
                key=f"download_analysis_{file_index}"
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
                "평균 신뢰도": "-",
                "최소 신뢰도": "-",
                "최대 신뢰도": "-",
                "YOLO 결과": "검사 오류",
                "Gemini 상태": "실행 안 됨"
            }
        )

    progress_bar.progress(
        (file_index + 1) / len(uploaded_files)
    )


status_text.success(
    "모든 이미지 검사가 완료되었습니다."
)


# ------------------------------------------------------------
# 전체 검사 요약
# ------------------------------------------------------------
st.divider()
st.header("전체 검사 요약")

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)

with summary_col1:
    st.metric(
        "검사 이미지 수",
        len(uploaded_files)
    )

with summary_col2:
    st.metric(
        "전체 탐지 개수",
        total_detected_count
    )

with summary_col3:
    st.metric(
        "Gemini 해석 완료",
        gemini_success_count
    )

with summary_col4:
    st.metric(
        "Gemini 해석 실패",
        gemini_failure_count
    )

if summary_rows:
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True
    )


st.caption(
    "Gemini AI 해석은 참고용입니다. 실제 정상/불량 판정은 "
    "검증된 검사 기준과 YOLO 모델 성능 평가 결과를 기준으로 결정해야 합니다."
)
