# 🏰 진격의 거인 - 거인 vs 인간 판별기 🏰

> **이미지를 업로드하면 거인인지 인간인지 판별해드립니다!**  
> Keras 모델을 활용한 이미지 분류 Streamlit 앱입니다.  
> 진격의 거인 팬들을 위한 재미있는 기능을 제공합니다.

---

## 📌 프로젝트 개요

이 프로젝트는 **딥러닝 기반의 이미지 분류 모델**을 사용하여 **거인**과 **인간**을 판별하는 웹 애플리케이션입니다.  
사용자는 **카메라 촬영** 또는 **파일 업로드** 방식을 선택하여 이미지를 입력하면, 모델이 해당 이미지가 거인인지 인간인지 판단합니다.  

본 프로젝트는 **Keras, TensorFlow, Streamlit**을 사용하여 구현되었습니다.

---

## 🚀 주요 기능

✅ **이미지 업로드 지원** (파일 업로드 또는 카메라 촬영)  
✅ **거인 vs 인간 판별** (AI 모델 활용)  
✅ **직관적인 UI** (Streamlit을 활용한 사용자 친화적인 인터페이스)  
✅ **분석 결과 시각화** (거인 감지 시 경고 이미지 출력)  
✅ **신뢰도(Confidence Score) 표시** (모델의 예측 신뢰도 제공)

---

## 🛠️ 설치 방법

### 1️⃣ **Python 3.10 환경 설정**  
**이 프로젝트는 Python 3.10 이하에서만 작동합니다.**  
아래 명령어를 실행하여 Python 3.10 가상 환경을 생성하세요.

```bash
conda create -n titan_detector python=3.10
conda activate titan_detector
