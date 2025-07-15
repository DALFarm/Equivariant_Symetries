import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "equivaraint-symmetries-for-inertial-navigation-systems/Simulation/Filters/Calibrated/"))
from SE23_se3_EqF import SE23_se3_EqF
import numpy as np

class EQF2:
    def __init__(self):
        self.filter = SE23_se3_EqF()
        self.t = None
    
    def predict(self, accel, gyro, dt):
        # SE23_se3_EqF는 propagate(t, vel, omega_noise, acc_noise, tau_noise, virtual_noise) 사용
        # 여기서는 t를 누적합으로 사용
        if self.t is None:
            self.t = 0.0
        else:
            self.t += dt
        # vel: [gyro, accel]을 (6,1)로 넘김
        vel = np.concatenate([gyro, accel]).reshape(6, 1)
        # 노이즈 파라미터는 임의의 작은 값 사용
        omega_noise = 0.01
        acc_noise = 0.01
        tau_noise = 0.001
        virtual_noise = 0.001
        self.filter.propagate(self.t, vel, omega_noise, acc_noise, tau_noise, virtual_noise)
    
    def update_gps(self, gps_pos):
        # 측정 업데이트: update(vel, omega_noise, acc_noise, tau_noise, virtual_noise, y, meas_noise, dt, propagate)
        # vel, dt 등은 predict에서와 동일하게 사용
        # 여기서는 propagate=False로 업데이트만 수행
        # dt는 임의의 작은 값 사용
        vel = np.zeros((6, 1))
        omega_noise = 0.01
        acc_noise = 0.01
        tau_noise = 0.001
        virtual_noise = 0.001
        meas_noise = 0.5
        dt = 0.01
        y = gps_pos.reshape(3, 1)
        self.filter.update(vel, omega_noise, acc_noise, tau_noise, virtual_noise, y, meas_noise, dt, propagate=False)
    
    def get_state(self):
        # getEstimate() -> (R, p, v, bw, ba, _, _)
        R, p, v, bw, ba, _, _ = self.filter.getEstimate()
        # R: 3x3 회전행렬, p: 3x1, v: 3x1
        # 회전행렬을 오일러 각으로 변환
        from scipy.spatial.transform import Rotation as Rot
        euler = Rot.from_matrix(R).as_euler('xyz', degrees=True)
        return {
            'position': p.flatten(),
            'velocity': v.flatten(),
            'euler': euler
        } 