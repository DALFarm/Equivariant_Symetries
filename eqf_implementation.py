#!/usr/bin/env python3
"""
EQF (Equivariant Filter) 구현 - 개선된 버전
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

class EQF:
    """Equivariant Filter (개선된 버전)"""
    def __init__(self):
        # 상태 벡터: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.state = np.zeros(9)
        self.state[2] = 0  # 초기 고도
        
        # 공분산 행렬 (EQF는 더 작은 초기 불확실성)
        self.P = np.eye(9) * 0.05
        
        # 프로세스 노이즈 (EQF는 더 작은 노이즈)
        self.Q = np.eye(9) * 0.005
        
        # 측정 노이즈 (EQF는 더 작은 측정 노이즈)
        self.R_gps = np.eye(3) * 0.5
        self.R_baro = np.array([[0.5]])
        
        # 중력 벡터
        self.g = np.array([0, 0, -9.81])
        
        # 초기화 플래그
        self.initialized = False
        
        # 자세 추정을 위한 쿼터니언
        self.quat = R.from_euler('xyz', [0, 0, 0], degrees=True)
    
    def predict(self, accel, gyro, dt):
        """예측 단계 - SO(3) 그룹 연산 활용"""
        if not self.initialized:
            self.initialized = True
            return
        
        # 현재 상태
        pos = self.state[0:3]
        vel = self.state[3:6]
        euler = self.state[6:9]
        
        # SO(3) 회전 행렬 계산
        R_body = self.quat.as_matrix()
        
        # 중력 보상 (Equivariant 방식)
        accel_corrected = R_body @ accel + self.g
        
        # 위치 및 속도 예측
        pos_new = pos + vel * dt + 0.5 * accel_corrected * dt**2
        vel_new = vel + accel_corrected * dt
        
        # 자세 예측 (SO(3) 곱셈)
        if np.linalg.norm(gyro) > 1e-6:
            delta_angle = gyro * dt
            delta_rot = R.from_rotvec(delta_angle)
            self.quat = self.quat * delta_rot
            euler_new = self.quat.as_euler('xyz', degrees=True)
        else:
            euler_new = euler
        
        # 상태 업데이트
        self.state[0:3] = pos_new
        self.state[3:6] = vel_new
        self.state[6:9] = euler_new
        
        # 자코비안 행렬 (Equivariant 방식)
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = np.eye(3) * dt * 0.1  # EQF는 더 작은 자세 영향
        
        # 공분산 예측
        self.P = F @ self.P @ F.T + self.Q * dt
    
    def update_gps(self, gps_pos):
        """GPS 측정 갱신 - Equivariant 방식"""
        if not self.initialized:
            return
        
        # 측정 예측
        h = self.state[0:3]
        
        # 자코비안 (Equivariant)
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        
        # 칼만 게인
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 상태 갱신
        innovation = gps_pos - h
        self.state = self.state + K @ innovation
        
        # 공분산 갱신
        I = np.eye(9)
        self.P = (I - K @ H) @ self.P
    
    def update_baro(self, baro_alt):
        """기압계 측정 갱신"""
        if not self.initialized:
            return
        
        # 측정 예측
        h = self.state[2]
        
        # 자코비안
        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        
        # 칼만 게인
        S = H @ self.P @ H.T + self.R_baro
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 상태 갱신
        innovation = baro_alt - h
        self.state = self.state + K.flatten() * innovation
        
        # 공분산 갱신
        I = np.eye(9)
        self.P = (I - K @ H) @ self.P
    
    def get_state(self):
        """현재 상태 반환"""
        return {
            'position': self.state[0:3],
            'velocity': self.state[3:6],
            'euler': self.state[6:9]
        } 