#!/usr/bin/env python3
"""
Groundtruth, PX4 EKF2, EKF, EQF 비교 스크립트
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyulog import ULog
from ekf_implementation import EKF
from eqf_implementation import EQF
from eqf2_implementation import EQF2
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

def load_ulog_data(ulog_file):
    """ULOG 파일에서 센서 데이터 로드"""
    print(f"Loading ULOG file: {ulog_file}")
    
    ulog = ULog(ulog_file)
    
    # IMU 데이터
    imu_data = ulog.get_dataset('vehicle_imu').data
    imu_timestamps = imu_data['timestamp'] / 1e6  # 마이크로초를 초로 변환
    
    # GPS 데이터
    gps_data = ulog.get_dataset('vehicle_gps_position').data
    gps_timestamps = gps_data['timestamp'] / 1e6
    # 위경도 -> local 변환 (초기값 기준 상대좌표)
    lat0 = gps_data['latitude_deg'][0]
    lon0 = gps_data['longitude_deg'][0]
    alt0 = gps_data['altitude_msl_m'][0]
    # 위도/경도 1도당 미터 근사값
    def latlon_to_m(lat, lon, lat0, lon0):
        dlat = (lat - lat0) * 111320
        dlon = (lon - lon0) * 40075000 * np.cos(np.radians(lat0)) / 360
        return dlat, dlon
    x_gps, y_gps = latlon_to_m(gps_data['latitude_deg'], gps_data['longitude_deg'], lat0, lon0)
    z_gps = gps_data['altitude_msl_m'] - alt0
    
    # 기압계 데이터
    baro_data = ulog.get_dataset('sensor_baro').data
    baro_timestamps = baro_data['timestamp'] / 1e6
    
    # 기압계 압력을 고도로 변환 (국제표준대기모델 사용)
    def pressure_to_altitude(pressure_pa, sea_level_pressure=101325.0):
        """압력을 고도로 변환 (국제표준대기모델)"""
        # 해수면 기준 압력 (Pa)
        P0 = sea_level_pressure
        # 중력가속도 (m/s^2)
        g = 9.80665
        # 공기 분자량 (kg/mol)
        M = 0.0289644
        # 기체상수 (J/(mol·K))
        R = 8.31432
        # 해수면 온도 (K)
        T0 = 288.15
        # 온도감률 (K/m)
        L = 0.0065
        
        # 고도 계산 (국제표준대기모델)
        altitude = T0 / L * (1 - (pressure_pa / P0) ** (R * L / (g * M)))
        return altitude
    
    # 초기 압력을 기준으로 상대 고도 계산
    initial_pressure = baro_data['pressure'][0]
    baro_altitudes = pressure_to_altitude(baro_data['pressure'], initial_pressure)
    
    # Groundtruth 데이터
    gt_data = ulog.get_dataset('vehicle_local_position_groundtruth').data
    gt_timestamps = gt_data['timestamp'] / 1e6
    
    # PX4 EKF2 데이터
    ekf2_data = ulog.get_dataset('estimator_states').data
    ekf2_timestamps = ekf2_data['timestamp'] / 1e6
    
    return {
        'imu': {
            'timestamps': imu_timestamps,
            'delta_velocity': np.column_stack([imu_data['delta_velocity[0]'], imu_data['delta_velocity[1]'], imu_data['delta_velocity[2]']]),
            'delta_angle': np.column_stack([imu_data['delta_angle[0]'], imu_data['delta_angle[1]'], imu_data['delta_angle[2]']]),
            'delta_velocity_dt': imu_data['delta_velocity_dt'],
            'delta_angle_dt': imu_data['delta_angle_dt']
        },
        'gps': {
            'timestamps': gps_timestamps,
            'position': np.column_stack([x_gps, y_gps, z_gps])
        },
        'baro': {
            'timestamps': baro_timestamps,
            'altitude': baro_altitudes,
            'pressure': baro_data['pressure']  # 임시로 압력 사용
        },
        'groundtruth': {
            'timestamps': gt_timestamps,
            'position': np.column_stack([gt_data['x'], gt_data['y'], gt_data['z']]),
            'velocity': np.column_stack([gt_data['vx'], gt_data['vy'], gt_data['vz']]),
            'euler': np.column_stack([gt_data['heading'], gt_data['heading'], gt_data['heading']])  # 간단한 근사
        },
        'px4_ekf2': {
            'timestamps': ekf2_timestamps,
            'position': np.column_stack([ekf2_data['states[7]'], ekf2_data['states[8]'], ekf2_data['states[9]']]),
            'velocity': np.column_stack([ekf2_data['states[10]'], ekf2_data['states[11]'], ekf2_data['states[12]']]),
            'euler': np.column_stack([ekf2_data['states[0]'], ekf2_data['states[1]'], ekf2_data['states[2]']])
        }
    }

def interpolate_data(target_timestamps, source_timestamps, source_data):
    """데이터를 타겟 타임스탬프에 맞춰 보간"""
    interpolated = np.zeros((len(target_timestamps), source_data.shape[1]))
    
    for i, t in enumerate(target_timestamps):
        # 가장 가까운 인덱스 찾기
        idx = np.argmin(np.abs(source_timestamps - t))
        interpolated[i] = source_data[idx]
    
    return interpolated

def run_ekf(imu_data, gps_data, baro_data):
    """EKF 실행"""
    print("Running EKF...")
    
    ekf = EKF()
    timestamps = imu_data['timestamps']
    
    # 결과 저장
    positions = []
    velocities = []
    eulers = []
    
    for i in range(len(timestamps)):
        # IMU 데이터 (delta 값을 속도/각속도로 변환)
        dt_vel = imu_data['delta_velocity_dt'][i]
        dt_ang = imu_data['delta_angle_dt'][i]
        
        if dt_vel > 0:
            accel = imu_data['delta_velocity'][i] / dt_vel
        else:
            accel = np.zeros(3)
            
        if dt_ang > 0:
            gyro = imu_data['delta_angle'][i] / dt_ang
        else:
            gyro = np.zeros(3)
        
        # 시간 간격 계산
        if i == 0:
            dt = 0.01  # 기본값
        else:
            dt = timestamps[i] - timestamps[i-1]
        
        # 예측
        ekf.predict(accel, gyro, dt)
        
        # GPS 갱신 (가장 가까운 GPS 데이터 찾기)
        gps_idx = np.argmin(np.abs(gps_data['timestamps'] - timestamps[i]))
        if abs(gps_data['timestamps'][gps_idx] - timestamps[i]) < 0.1:  # 100ms 이내
            gps_pos = gps_data['position'][gps_idx]
            ekf.update_gps(gps_pos)
        
        # 기압계 갱신 (활성화)
        baro_idx = np.argmin(np.abs(baro_data['timestamps'] - timestamps[i]))
        if abs(baro_data['timestamps'][baro_idx] - timestamps[i]) < 0.1:  # 100ms 이내
            baro_alt = baro_data['altitude'][baro_idx]
            ekf.update_baro(baro_alt)
        
        # 상태 저장
        state = ekf.get_state()
        positions.append(state['position'])
        velocities.append(state['velocity'])
        eulers.append(state['euler'])
    
    return {
        'timestamps': timestamps,
        'position': np.array(positions),
        'velocity': np.array(velocities),
        'euler': np.array(eulers)
    }

def run_eqf(imu_data, gps_data, baro_data):
    """EQF 실행"""
    print("Running EQF...")
    eqf = EQF()
    timestamps = imu_data['timestamps']
    positions = []
    velocities = []
    eulers = []
    
    for i in range(len(timestamps)):
        dt_vel = imu_data['delta_velocity_dt'][i]
        dt_ang = imu_data['delta_angle_dt'][i]
        if dt_vel > 0:
            accel = imu_data['delta_velocity'][i] / dt_vel
        else:
            accel = np.zeros(3)
        if dt_ang > 0:
            gyro = imu_data['delta_angle'][i] / dt_ang
        else:
            gyro = np.zeros(3)
        if i == 0:
            dt = 0.01
        else:
            dt = timestamps[i] - timestamps[i-1]
        eqf.predict(accel, gyro, dt)
        gps_idx = np.argmin(np.abs(gps_data['timestamps'] - timestamps[i]))
        if abs(gps_data['timestamps'][gps_idx] - timestamps[i]) < 0.1:
            gps_pos = gps_data['position'][gps_idx]
            eqf.update_gps(gps_pos)
        # 기압계 갱신 (활성화)
        baro_idx = np.argmin(np.abs(baro_data['timestamps'] - timestamps[i]))
        if abs(baro_data['timestamps'][baro_idx] - timestamps[i]) < 0.1:
            baro_alt = baro_data['altitude'][baro_idx]
            eqf.update_baro(baro_alt)
        state = eqf.get_state()
        positions.append(state['position'])
        velocities.append(state['velocity'])
        eulers.append(state['euler'])
    return {
        'timestamps': timestamps,
        'position': np.array(positions),
        'velocity': np.array(velocities),
        'euler': np.array(eulers)
    }

def run_eqf2(imu_data, gps_data, baro_data):
    """EQF2 실행"""
    print("Running EQF2...")
    eqf2 = EQF2()
    timestamps = imu_data['timestamps']
    positions = []
    velocities = []
    eulers = []
    for i in range(len(timestamps)):
        dt_vel = imu_data['delta_velocity_dt'][i]
        dt_ang = imu_data['delta_angle_dt'][i]
        if dt_vel > 0:
            accel = imu_data['delta_velocity'][i] / dt_vel
        else:
            accel = np.zeros(3)
        if dt_ang > 0:
            gyro = imu_data['delta_angle'][i] / dt_ang
        else:
            gyro = np.zeros(3)
        if i == 0:
            dt = 0.01
        else:
            dt = timestamps[i] - timestamps[i-1]
        eqf2.predict(accel, gyro, dt)
        gps_idx = np.argmin(np.abs(gps_data['timestamps'] - timestamps[i]))
        if abs(gps_data['timestamps'][gps_idx] - timestamps[i]) < 0.1:
            gps_pos = gps_data['position'][gps_idx]
            eqf2.update_gps(gps_pos)
        # EQF2는 기압계 갱신을 지원하지 않음 (GPS만 사용)
        state = eqf2.get_state()
        positions.append(state['position'])
        velocities.append(state['velocity'])
        eulers.append(state['euler'])
    return {
        'timestamps': timestamps,
        'position': np.array(positions),
        'velocity': np.array(velocities),
        'euler': np.array(eulers)
    }

def calculate_metrics(estimated, groundtruth):
    """성능 메트릭 계산"""
    # Groundtruth를 추정 데이터 타임스탬프에 맞춰 보간
    gt_interpolated = interpolate_data(
        estimated['timestamps'], 
        groundtruth['timestamps'], 
        groundtruth['position']
    )
    
    # 오차 계산
    errors = estimated['position'] - gt_interpolated
    
    # RMSE 계산
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    
    # MAE 계산
    mae = np.mean(np.abs(errors), axis=0)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'errors': errors,
        'groundtruth_interpolated': gt_interpolated
    }

def plot_comparison(groundtruth, px4_ekf2, ekf_results, eqf_results, eqf2_results, metrics_ekf, metrics_px4, metrics_eqf, metrics_eqf2):
    """결과 플롯 (EQF2 포함, 2x3 subplot)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Groundtruth vs PX4_EKF2 vs EKF vs EQF vs EQF2 Comparison', fontsize=16)
    # EKF2 위치를 EKF 타임스탬프에 보간
    # linter: fill_value must be float, so use 0.0 (will only be used if out of bounds)
    ekf2_interp_x = interp1d(px4_ekf2['timestamps'], px4_ekf2['position'][:, 0], bounds_error=False, fill_value=0.0)(ekf_results['timestamps'])
    ekf2_interp_y = interp1d(px4_ekf2['timestamps'], px4_ekf2['position'][:, 1], bounds_error=False, fill_value=0.0)(ekf_results['timestamps'])
    ekf2_interp_z = interp1d(px4_ekf2['timestamps'], px4_ekf2['position'][:, 2], bounds_error=False, fill_value=0.0)(ekf_results['timestamps'])
    # X Position
    axes[0, 0].plot(ekf_results['timestamps'], metrics_ekf['groundtruth_interpolated'][:, 0], 'k-', label='Groundtruth', linewidth=2)
    axes[0, 0].plot(ekf_results['timestamps'], ekf2_interp_x, 'b-', label='PX4_EKF2', linewidth=2)
    axes[0, 0].plot(ekf_results['timestamps'], ekf_results['position'][:, 0], 'r-', label='EKF', linewidth=2)
    axes[0, 0].plot(ekf_results['timestamps'], eqf_results['position'][:, 0], 'g-', label='EQF', linewidth=2)
    axes[0, 0].plot(ekf_results['timestamps'], eqf2_results['position'][:, 0], 'm-', label='EQF2', linewidth=2)
    axes[0, 0].set_title('X Position')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # Y Position
    axes[0, 1].plot(ekf_results['timestamps'], metrics_ekf['groundtruth_interpolated'][:, 1], 'k-', label='Groundtruth', linewidth=2)
    axes[0, 1].plot(ekf_results['timestamps'], ekf2_interp_y, 'b-', label='PX4_EKF2', linewidth=2)
    axes[0, 1].plot(ekf_results['timestamps'], ekf_results['position'][:, 1], 'r-', label='EKF', linewidth=2)
    axes[0, 1].plot(ekf_results['timestamps'], eqf_results['position'][:, 1], 'g-', label='EQF', linewidth=2)
    axes[0, 1].plot(ekf_results['timestamps'], eqf2_results['position'][:, 1], 'm-', label='EQF2', linewidth=2)
    axes[0, 1].set_title('Y Position')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    # Z Position
    axes[0, 2].plot(ekf_results['timestamps'], metrics_ekf['groundtruth_interpolated'][:, 2], 'k-', label='Groundtruth', linewidth=2)
    axes[0, 2].plot(ekf_results['timestamps'], ekf2_interp_z, 'b-', label='PX4_EKF2', linewidth=2)
    axes[0, 2].plot(ekf_results['timestamps'], ekf_results['position'][:, 2], 'r-', label='EKF', linewidth=2)
    axes[0, 2].plot(ekf_results['timestamps'], eqf_results['position'][:, 2], 'g-', label='EQF', linewidth=2)
    axes[0, 2].plot(ekf_results['timestamps'], eqf2_results['position'][:, 2], 'm-', label='EQF2', linewidth=2)
    axes[0, 2].set_title('Z Position')
    axes[0, 2].set_ylabel('Position (m)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    # X Position Error
    axes[1, 0].plot(ekf_results['timestamps'], metrics_ekf['errors'][:, 0], 'r-', label='EKF Error', linewidth=2)
    axes[1, 0].plot(px4_ekf2['timestamps'], metrics_px4['errors'][:, 0], 'b-', label='PX4_EKF2 Error', linewidth=2)
    axes[1, 0].plot(eqf_results['timestamps'], metrics_eqf['errors'][:, 0], 'g-', label='EQF Error', linewidth=2)
    axes[1, 0].plot(eqf2_results['timestamps'], metrics_eqf2['errors'][:, 0], 'y-', label='EQF2 Error', linewidth=2)
    axes[1, 0].set_title('X Position Error')
    axes[1, 0].set_ylabel('Error (m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    # Y Position Error
    axes[1, 1].plot(ekf_results['timestamps'], metrics_ekf['errors'][:, 1], 'r-', label='EKF Error', linewidth=2)
    axes[1, 1].plot(px4_ekf2['timestamps'], metrics_px4['errors'][:, 1], 'b-', label='PX4_EKF2 Error', linewidth=2)
    axes[1, 1].plot(eqf_results['timestamps'], metrics_eqf['errors'][:, 1], 'g-', label='EQF Error', linewidth=2)
    axes[1, 1].plot(eqf2_results['timestamps'], metrics_eqf2['errors'][:, 1], 'y-', label='EQF2 Error', linewidth=2)
    axes[1, 1].set_title('Y Position Error')
    axes[1, 1].set_ylabel('Error (m)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    # Z Position Error
    axes[1, 2].plot(ekf_results['timestamps'], metrics_ekf['errors'][:, 2], 'r-', label='EKF Error', linewidth=2)
    axes[1, 2].plot(px4_ekf2['timestamps'], metrics_px4['errors'][:, 2], 'b-', label='PX4_EKF2 Error', linewidth=2)
    axes[1, 2].plot(eqf_results['timestamps'], metrics_eqf['errors'][:, 2], 'g-', label='EQF Error', linewidth=2)
    axes[1, 2].plot(eqf2_results['timestamps'], metrics_eqf2['errors'][:, 2], 'y-', label='EQF2 Error', linewidth=2)
    axes[1, 2].set_title('Z Position Error')
    axes[1, 2].set_ylabel('Error (m)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 2D Trajectory Plot (xy 평면, 파일 저장)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 7))
    # linter: ensure ax3 is always an Axes object
    if isinstance(ax3, np.ndarray):
        ax3 = ax3.flatten()[0]
    ax3.plot(metrics_ekf['groundtruth_interpolated'][:, 0], metrics_ekf['groundtruth_interpolated'][:, 1], 'k-', label='Groundtruth', linewidth=3)
    ax3.plot(ekf_results['position'][:, 0], ekf_results['position'][:, 1], 'r-', label='EKF', linewidth=2)
    ax3.plot(eqf_results['position'][:, 0], eqf_results['position'][:, 1], 'g-', label='EQF', linewidth=2)
    ax3.plot(eqf2_results['position'][:, 0], eqf2_results['position'][:, 1], 'm-', label='EQF2', linewidth=2)
    # EKF2는 보간된 x, y 사용 (2x3 plot과 동일)
    ax3.plot(ekf2_interp_x, ekf2_interp_y, 'b-', label='PX4_EKF2', linewidth=2)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('2D Trajectory Comparison')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    plt.tight_layout()
    plt.savefig('trajectory_comparison_2d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3D Trajectory Plot (별도 figure, 파일 저장+show)
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    # Groundtruth
    ax2.plot(metrics_ekf['groundtruth_interpolated'][:, 0], metrics_ekf['groundtruth_interpolated'][:, 1], metrics_ekf['groundtruth_interpolated'][:, 2], 'k-', label='Groundtruth', linewidth=3)
    # EKF
    ax2.plot(ekf_results['position'][:, 0], ekf_results['position'][:, 1], ekf_results['position'][:, 2], 'r-', label='EKF', linewidth=2)
    # EQF
    ax2.plot(eqf_results['position'][:, 0], eqf_results['position'][:, 1], eqf_results['position'][:, 2], 'g-', label='EQF', linewidth=2)
    # EQF2
    ax2.plot(eqf2_results['position'][:, 0], eqf2_results['position'][:, 1], eqf2_results['position'][:, 2], 'm-', label='EQF2', linewidth=2)
    # PX4_EKF2
    ax2.plot(px4_ekf2['position'][:, 0], px4_ekf2['position'][:, 1], px4_ekf2['position'][:, 2], 'b-', label='PX4_EKF2', linewidth=2)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_zlabel('Z Position (m)')
    ax2.set_title('3D Trajectory Comparison')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    """메인 함수"""
    # ULOG 파일 경로
    ulog_file = "0707_SITL/log_4_2025-7-7-09-05-20.ulg"
    
    if not os.path.exists(ulog_file):
        print(f"Error: ULOG file not found: {ulog_file}")
        return
    
    # 데이터 로드
    data = load_ulog_data(ulog_file)
    
    # EKF 실행
    ekf_results = run_ekf(data['imu'], data['gps'], data['baro'])
    # EQF 실행
    eqf_results = run_eqf(data['imu'], data['gps'], data['baro'])
    # EQF2 실행
    eqf2_results = run_eqf2(data['imu'], data['gps'], data['baro'])
    
    # 성능 메트릭 계산
    metrics_ekf = calculate_metrics(ekf_results, data['groundtruth'])
    metrics_px4 = calculate_metrics(data['px4_ekf2'], data['groundtruth'])
    metrics_eqf = calculate_metrics(eqf_results, data['groundtruth'])
    metrics_eqf2 = calculate_metrics(eqf2_results, data['groundtruth'])
    
    # 결과 출력
    print("\n=== Performance Metrics ===")
    print("EKF RMSE (X, Y, Z):", metrics_ekf['rmse'])
    print("EKF MAE (X, Y, Z):", metrics_ekf['mae'])
    print("PX4_EKF2 RMSE (X, Y, Z):", metrics_px4['rmse'])
    print("PX4_EKF2 MAE (X, Y, Z):", metrics_px4['mae'])
    print("EQF RMSE (X, Y, Z):", metrics_eqf['rmse'])
    print("EQF MAE (X, Y, Z):", metrics_eqf['mae'])
    print("EQF2 RMSE (X, Y, Z):", metrics_eqf2['rmse'])
    print("EQF2 MAE (X, Y, Z):", metrics_eqf2['mae'])
    
    # 결과 플롯
    plot_comparison(data['groundtruth'], data['px4_ekf2'], ekf_results, eqf_results, eqf2_results, metrics_ekf, metrics_px4, metrics_eqf, metrics_eqf2)
    
    # 결과를 CSV로 저장
    px4_interp = interpolate_data(ekf_results['timestamps'], data['px4_ekf2']['timestamps'], data['px4_ekf2']['position'])
    eqf_interp = interpolate_data(ekf_results['timestamps'], eqf_results['timestamps'], eqf_results['position'])
    eqf2_interp = interpolate_data(ekf_results['timestamps'], eqf2_results['timestamps'], eqf2_results['position'])
    gt_interp = metrics_ekf['groundtruth_interpolated']
    results_df = pd.DataFrame({
        'timestamp': ekf_results['timestamps'],
        'ekf_x': ekf_results['position'][:, 0],
        'ekf_y': ekf_results['position'][:, 1],
        'ekf_z': ekf_results['position'][:, 2],
        'eqf_x': eqf_interp[:, 0],
        'eqf_y': eqf_interp[:, 1],
        'eqf_z': eqf_interp[:, 2],
        'eqf2_x': eqf2_interp[:, 0],
        'eqf2_y': eqf2_interp[:, 1],
        'eqf2_z': eqf2_interp[:, 2],
        'px4_ekf2_x': px4_interp[:, 0],
        'px4_ekf2_y': px4_interp[:, 1],
        'px4_ekf2_z': px4_interp[:, 2],
        'groundtruth_x': gt_interp[:, 0],
        'groundtruth_y': gt_interp[:, 1],
        'groundtruth_z': gt_interp[:, 2]
    })
    results_df.to_csv('comparison_results.csv', index=False)
    print("\nResults saved to comparison_results.csv")

if __name__ == "__main__":
    main() 