# 산학협력 프로젝트 & 졸업작품: 수면모니터링 딥러닝 모델 개발
## Non-contact sleep stage classification based on CNN–LSTM model using UWB radar
## UWB 레이더를 이용한 CNN–LSTM 모델 기반 비접촉식 수면 단계 분류

### CNN-LSTM 모델로 2023 대한의용생체공학회에 참가함.
### 전체 정확도는 55.906% 이고, test loss는 0.085이다.
### class별 정확도는 N1+N2 단계는 72.301%, N3 단계는 24.436%, wake 단계는 51.087%, REM 단계는 32.476% 이다.
total | N1+N2 | N3 | wake | REM
--- | --- | --- | --- | --- 
55.906% | 72.301% | 24.436% | 51.087% | 32.476%

### 수면데이터 전처리 과정
### 1. UnionData.m 파일을 사용해 dynamic threshold를 수행
### 2. 모든 subject 데이터를 합쳐서 normalize 수행
### 3. 다시 subject 별로 데이터를 분할
### 4. dataPreprocessing.m 파일을 사용해 CWT 이미지화 수행
