# Equivariant Symmetries - EKF/EKF2/EqF/EqF2 비교 분석

## 프로젝트 개요

이 프로젝트는 다양한 확장 칼만 필터(Extended Kalman Filter) 알고리즘들을 비교 분석하기 위한 코드입니다.

## 주요 내용

### 비교 대상 알고리즘
- **EKF**: 기존 확장 칼만 필터
- **EKF2**: 개선된 확장 칼만 필터
- **EqF**: 등변 확장 칼만 필터 (Equivariant Extended Kalman Filter)
- **EqF2**: 개선된 등변 확장 칼만 필터

### 데이터 소스
- PX4 비행제어기에서 생성된 ULOG 파일에서 센서 데이터를 추출
- 추출된 센서 데이터를 사용하여 각 알고리즘의 성능을 비교 분석

## 파일 구조

- `compare_groundtruth_ekf_px4.py`: EKF와 ground truth 비교 분석
- `ekf_implementation.py`: EKF 구현
- `eqf_implementation.py`: EqF 구현
- `eqf2_implementation.py`: EqF2 구현
- `px4_replay_import.m`: PX4 ULOG 데이터 import를 위한 MATLAB 스크립트
- `pylie/`: Lie 그룹 연산을 위한 Python 라이브러리

## 주의사항

⚠️ **이 코드는 완벽한 구현이 아닙니다.** 연구 및 비교 분석 목적으로 작성되었으며, 실제 운영 환경에서 사용하기 전에 충분한 검증이 필요합니다.

## 사용법

1. PX4 ULOG 파일을 준비합니다
2. `px4_replay_import.m`을 사용하여 센서 데이터를 추출합니다
3. 각 알고리즘 구현 파일을 실행하여 성능을 비교합니다

## 의존성

필요한 패키지는 `requirements.txt` 파일을 참조하세요.

## 라이선스

이 프로젝트는 연구 목적으로 작성되었습니다. 