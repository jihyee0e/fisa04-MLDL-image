# 3.10 버전 이하에서만 작동합니다.
# conda create -n test2 python=3.10
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL

# 📌 앱 설정
st.set_page_config(page_title="진격의 거인 - 거인 vs 인간 판별기", layout="centered")

# 🏰 앱 제목 및 설명
st.title("🛡️ 진격의 거인 - 거인 vs 인간 판별기")
st.markdown("### **이미지를 업로드하면 거인인지 인간인지 판별해드립니다.**")
st.write("📌 카메라 촬영 또는 파일 업로드 중 하나를 선택하세요!")

# 🖼️ 배너 이미지 추가 (로컬 파일)
st.image("titan_vs_human_banner.png", use_column_width=True)

# 🏗️ 모델 로드
@st.cache_resource
def load_keras_model():
    return load_model('keras_model.h5', compile=False)

model = load_keras_model()

# 📄 레이블 로드
class_names = open('labels.txt', 'r', encoding='utf-8').readlines()

# 🖼️ 이미지 입력 방식 선택
input_method = st.radio("이미지 입력 방식", ["📷 카메라 촬영", "📁 파일 업로드"])

# 📷 카메라 또는 파일 업로드
if input_method == "📷 카메라 촬영":
    img_file_buffer = st.camera_input("정중앙에 사물을 위치하고 '촬영' 버튼을 누르세요.")
else:
    img_file_buffer = st.file_uploader("이미지 파일 업로드", type=["png", "jpg", "jpeg"])

# 📌 이미지가 있으면 처리
if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("RGB")

    # 🎨 이미지 미리보기
    st.image(image, caption="🔍 업로드한 이미지", use_column_width=True)

    # 🎯 분석 버튼 추가
    if st.button("🔍 분석하기"):
        # ✅ 이미지 전처리 (224x224 변환 및 정규화)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # 🏗️ 모델 예측 수행
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # 📌 예측 결과 출력
        st.markdown("## 🏆 결과")
        if "거인" in class_name:
            st.markdown(f"### 🟥 **거인입니다!** (⚠️ 도망쳐!)")
            st.image("titan_alert.png", caption="⚠️ 거인 감지됨!", use_column_width=True)  # 거인 이미지 추가
        else:
            st.markdown(f"### 🟩 **인간입니다!** (✅ 안전합니다!)")
            st.image("human_safe.png", caption="✅ 인간 감지됨!", use_column_width=True)  # 인간 이미지 추가

        st.write(f"**🔍 예측 확신도:** {confidence_score:.2%}")

        # 🎭 추가 시각적 효과
        st.progress(float(confidence_score))

# 📝 푸터
st.markdown("---")