import numpy as np
from scipy import special
import matplotlib.pyplot as plt
 
# Parameters
sample_rate = 20e6  # 20 MHz sample rate
hop_rate = 1000  # 1000 hops per second
hop_duration = 1.0 / hop_rate  # 0.001 seconds per hop
total_duration = 0.06  # 1 second total duration
num_hops = int(total_duration * hop_rate)
samples_per_hop = int(sample_rate * hop_duration)

# Define channel frequencies
# channel_freqs = np.array([-7.5e6, -4.5e6, -1.5e6, 1.5e6, 4.5e6, 7.5e6])
# channel_freqs = np.array([ -4.5e6,-3e6, -1.5e6, 1.5e6,2.5e6,3.5e6, 4.5e6])
channel_freqs = np.array([0])

# Generate a pseudo-random hop sequence
np.random.seed(42)  # For reproducibility
hop_sequence = np.random.choice(channel_freqs, size=num_hops, replace=True)
# hop_sequence = np.tile(np.array([-7.5e6, -2.5e6, 2.5e6, 7.5e6]), 10)

# Time vector for one hop
t_hop = np.linspace(0, hop_duration, samples_per_hop, endpoint=False)

# Generate the complex baseband signal with single sinc wave per hop
signal = np.zeros(num_hops * samples_per_hop, dtype=np.complex64)
for i in range(num_hops):
    f_center = hop_sequence[i]
    start_idx = i * samples_per_hop
    end_idx = start_idx + samples_per_hop
    
    # 生成單個sinc波 - 時域上就是一個sinc函數
    bandwidth = 10e6  # 3MHz頻寬
    num_sinc_waves = 50  # 每個hop內使用5個sinc波
    sinc_signal = np.zeros_like(t_hop, dtype=np.complex64)
    
    # 在hop時間內均勻分佈多個sinc波
    for k in range(num_sinc_waves):
        # 計算每個sinc波的時間偏移
        offset = (k - num_sinc_waves//2) * hop_duration / num_sinc_waves
        t_offset = t_hop - hop_duration/2 + offset
        
        # 生成sinc波
        sinc_component = np.sinc(bandwidth * t_offset)
        
        # 加入隨機幅度和相位變化
        amplitude = np.random.uniform(0.7, 0.8)
        phase = np.random.uniform(0, 2*np.pi)
        # amplitude = 2
        # phase = np.pi
        
        # 組合多個sinc波
        sinc_signal += amplitude * sinc_component * np.exp(1j * phase)
    
    # 加入載波頻率
    carrier = 1*np.exp(1j * 2 * np.pi * f_center * t_hop)
    hop_signal = sinc_signal * carrier
    
    signal[start_idx:end_idx] = hop_signal * np.hanning(samples_per_hop)

# Normalize the signal
signal /= np.max(np.abs(signal))

# # 觀察單個sinc波的時域信號
# print("=== Observing Single Sinc Wave Time Domain Signal ===")
# # 使用第一個hop的sinc信號進行觀察
# first_hop_signal = signal[:samples_per_hop]
# time_axis = np.arange(len(first_hop_signal)) / sample_rate * 1e6  # 轉換為微秒

# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# fig.suptitle('Single Sinc Wave Analysis', fontsize=14)

# # 時域信號 - 幅度
# axes[0, 0].plot(time_axis, np.abs(first_hop_signal), 'b-', linewidth=1)
# axes[0, 0].set_title('Time Domain Signal (Sinc Wave Amplitude)')
# axes[0, 0].set_xlabel('Time (μs)')
# axes[0, 0].set_ylabel('Amplitude')
# axes[0, 0].grid(True, alpha=0.3)

# # 時域信號 - 實部
# axes[0, 1].plot(time_axis, np.real(first_hop_signal), 'r-', linewidth=1)
# axes[0, 1].set_title('Time Domain Signal (Real Part)')
# axes[0, 1].set_xlabel('Time (μs)')
# axes[0, 1].set_ylabel('Amplitude')
# axes[0, 1].grid(True, alpha=0.3)

# # 時域信號 - 虛部
# axes[1, 0].plot(time_axis, np.imag(first_hop_signal), 'g-', linewidth=1)
# axes[1, 0].set_title('Time Domain Signal (Imaginary Part)')
# axes[1, 0].set_xlabel('Time (μs)')
# axes[1, 0].set_ylabel('Amplitude')
# axes[1, 0].grid(True, alpha=0.3)

# # 頻域信號
# fft_sinc = np.fft.fft(first_hop_signal)
# freq_axis = np.fft.fftfreq(len(first_hop_signal), 1/sample_rate)
# freq_shifted = np.fft.fftshift(freq_axis)
# fft_shifted = np.fft.fftshift(fft_sinc)

# axes[1, 1].plot(freq_shifted/1e6, np.abs(fft_shifted), 'purple', linewidth=1)
# axes[1, 1].set_title('Frequency Domain - Magnitude Spectrum')
# axes[1, 1].set_xlabel('Frequency (MHz)')
# axes[1, 1].set_ylabel('Magnitude')
# axes[1, 1].grid(True, alpha=0.3)
# # 顯示完整頻譜（負頻到正頻）
# axes[1, 1].set_xlim(-sample_rate/2/1e6, sample_rate/2/1e6)  # -10MHz 到 +10MHz

# plt.tight_layout()
# plt.savefig('single_sinc_wave_analysis.png', dpi=150, bbox_inches='tight')
# plt.show()
# plt.close()

# print("Single sinc wave analysis saved as 'single_sinc_wave_analysis.png'")

# # FFT分析 - 檢查sinc波信號特性
# print("=== Sinc Wave Signal Analysis ===")

# # 1. 檢查整體信號的頻譜
# print(f"Signal length: {len(signal)} samples")
# print(f"Signal duration: {len(signal)/sample_rate:.3f} seconds")
# print(f"Signal max amplitude: {np.max(np.abs(signal)):.6f}")
# print(f"Signal average power: {np.mean(np.abs(signal)**2):.6f}")

# # 2. 分析前幾個hop的頻譜
# print("\n=== First 5 Hops Spectrum Analysis ===")
# for i in range(min(5, num_hops)):
#     start_idx = i * samples_per_hop
#     end_idx = start_idx + samples_per_hop
#     hop_signal = signal[start_idx:end_idx]
    
#     # FFT分析
#     fft_hop = np.fft.fft(hop_signal)
#     freq_axis = np.fft.fftfreq(samples_per_hop, 1/sample_rate)
#     freq_shifted = np.fft.fftshift(freq_axis)
#     fft_shifted = np.fft.fftshift(fft_hop)
    
#     # 找到主要頻率成分
#     power_spectrum = np.abs(fft_shifted)**2
#     max_power_idx = np.argmax(power_spectrum)
#     main_freq = freq_shifted[max_power_idx]
    
#     print(f"Hop {i}: Center frequency = {hop_sequence[i]/1e6:.1f} MHz")
#     print(f"  Main frequency component: {main_freq/1e6:.2f} MHz")
#     print(f"  Signal power: {np.sum(power_spectrum):.2f}")

# def calculate_bandwidth_3db(freq_axis, power_spectrum):
#     """Calculate -3dB bandwidth"""
#     max_power = np.max(power_spectrum)
#     threshold = max_power / 2  # -3dB
    
#     above_threshold = power_spectrum > threshold
#     if np.any(above_threshold):
#         indices = np.where(above_threshold)[0]
#         bandwidth = freq_axis[indices[-1]] - freq_axis[indices[0]]
#         return bandwidth
#     return 0

# # 為每個hop繪製完整的頻譜圖
# print("\n=== Generating Individual Hop Sinc Wave Spectrum Plots ===")

# # 分析前10個hop
# for hop_idx in range(min(10, num_hops)):
#     start_idx = hop_idx * samples_per_hop
#     end_idx = start_idx + samples_per_hop
#     hop_signal = signal[start_idx:end_idx]
    
#     # FFT分析
#     fft_hop = np.fft.fft(hop_signal)
#     freq_axis = np.fft.fftfreq(samples_per_hop, 1/sample_rate)
#     freq_shifted = np.fft.fftshift(freq_axis)
#     fft_shifted = np.fft.fftshift(fft_hop)
    
#     # 計算功率譜密度
#     power_spectrum = np.abs(fft_shifted)**2
#     power_spectrum_db = 10 * np.log10(power_spectrum + 1e-12)
    
#     # 繪製完整的頻譜圖
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle(f'Hop {hop_idx} - Sinc Wave Spectrum Analysis (Center: {hop_sequence[hop_idx]/1e6:.1f} MHz)', fontsize=14)
    
#     # 時域信號 - 幅度
#     time_axis = np.arange(len(hop_signal)) / sample_rate * 1e6  # 轉換為微秒
#     amplitude_signal = np.abs(hop_signal)
#     axes[0, 0].plot(time_axis, amplitude_signal, 'b-', linewidth=1)
#     axes[0, 0].set_title('Time Domain Signal (Sinc Wave Amplitude)')
#     axes[0, 0].set_xlabel('Time (μs)')
#     axes[0, 0].set_ylabel('Amplitude')
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # 頻域信號 - 幅度譜
#     axes[0, 1].plot(freq_shifted/1e6, np.abs(fft_shifted), 'g-', linewidth=1)
#     axes[0, 1].set_title('Frequency Domain - Magnitude Spectrum')
#     axes[0, 1].set_xlabel('Frequency (MHz)')
#     axes[0, 1].set_ylabel('Magnitude')
#     axes[0, 1].grid(True, alpha=0.3)
#     # 顯示完整頻譜（負頻到正頻）
#     axes[0, 1].set_xlim(-sample_rate/2/1e6, sample_rate/2/1e6)  # -10MHz 到 +10MHz
    
#     # 功率譜密度 (dB)
#     axes[1, 0].plot(freq_shifted/1e6, power_spectrum_db, 'purple', linewidth=1)
#     axes[1, 0].set_title('Power Spectral Density (dB)')
#     axes[1, 0].set_xlabel('Frequency (MHz)')
#     axes[1, 0].set_ylabel('Power (dB)')
#     axes[1, 0].grid(True, alpha=0.3)
#     # 顯示完整頻譜（負頻到正頻）
#     axes[1, 0].set_xlim(-sample_rate/2/1e6, sample_rate/2/1e6)  # -10MHz 到 +10MHz
      
#     plt.tight_layout()
#     plt.savefig(f'sinc_hop_{hop_idx}_complete_spectrum.png', dpi=150, bbox_inches='tight')
#     plt.show()
    
#     print(f"Sinc Hop {hop_idx} complete spectrum saved as 'sinc_hop_{hop_idx}_complete_spectrum.png'")

# # 額外：繪製所有hop的頻譜對比圖
# print("\n=== Generating All Sinc Hops Spectrum Comparison ===")
# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# fig.suptitle('First 10 Sinc Hops - Spectrum Comparison', fontsize=16)

# for hop_idx in range(min(10, num_hops)):
#     row = hop_idx // 5
#     col = hop_idx % 5
#     ax = axes[row, col]
    
#     start_idx = hop_idx * samples_per_hop
#     end_idx = start_idx + samples_per_hop
#     hop_signal = signal[start_idx:end_idx]
    
#     # FFT分析
#     fft_hop = np.fft.fft(hop_signal)
#     freq_axis = np.fft.fftfreq(samples_per_hop, 1/sample_rate)
#     freq_shifted = np.fft.fftshift(freq_axis)
#     fft_shifted = np.fft.fftshift(fft_hop)
    
#     # 繪製頻譜
#     ax.plot(freq_shifted/1e6, np.abs(fft_shifted), 'b-', linewidth=1)
#     ax.set_title(f'Sinc Hop {hop_idx}\nCenter: {hop_sequence[hop_idx]/1e6:.1f} MHz')
#     ax.set_xlabel('Frequency (MHz)')
#     ax.set_ylabel('Magnitude')
#     ax.grid(True, alpha=0.3)
    
#     # 顯示完整頻譜（負頻到正頻）
#     ax.set_xlim(-sample_rate/2/1e6, sample_rate/2/1e6)  # -10MHz 到 +10MHz

# plt.tight_layout()
# plt.savefig('all_sinc_hops_spectrum_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()

# print("All sinc hops spectrum comparison saved as 'all_sinc_hops_spectrum_comparison.png'")

# print("\n=== STFT Analysis ===")
# from scipy import signal as scipy_signal

# # 使用原始複數信號進行STFT分析
# print("Performing STFT on the complex signal...")

# # STFT參數
# window_length = 1500  # 窗長
# overlap = window_length*0.3  # 重疊長度
# nperseg = window_length
# noverlap = overlap
# # sample_rate = 160e6
# print(f"總訊號長度{len(signal)}")
# # 執行STFT
# f_stft, t_stft, Zxx = scipy_signal.stft(signal, fs=20e6,window='hann',
#                                        nperseg=nperseg, noverlap=noverlap,return_onesided=False)

# # 繪製STFT時頻圖
# plt.figure(figsize=(12, 8))

# # 時頻圖
# # 推薦的解決方案
# T, F = np.meshgrid(t_stft*1000, f_stft/1e6)
# plt.contourf(T, F, np.abs(Zxx), levels=50, cmap='viridis')
# plt.title('STFT - Time-Frequency Analysis')
# plt.xlabel('Time (ms)')
# plt.ylabel('Frequency (MHz)')
# plt.colorbar(label='Magnitude')

# plt.tight_layout()
# plt.savefig('stft_analysis.png', dpi=150, bbox_inches='tight')
# plt.show()
# plt.close()

# print("STFT analysis saved as 'stft_analysis.png'")

# # 額外：繪製STFT的3D視圖
# print("Generating 3D STFT visualization...")
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 創建網格
# T, F = np.meshgrid(t_stft*1000, f_stft/1e6)
# Z = np.abs(Zxx)

# # 3D表面圖
# surf = ax.plot_surface(T, F, Z, cmap='viridis', alpha=0.8)
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Frequency (MHz)')
# ax.set_zlabel('Magnitude')
# ax.set_title('3D STFT - Time-Frequency Analysis')
# # ax.view_init(elev=90, azim=0)

# plt.tight_layout()
# plt.savefig('stft_3d_analysis.png', dpi=150, bbox_inches='tight')
# plt.show()
# plt.close()

# print("3D STFT analysis saved as 'stft_3d_analysis.png'")

# # 繪製signal的時域幅度
# print("\n=== Plotting Signal Time Domain Amplitude ===")

# # 時域幅度分析
# time_axis = np.arange(len(signal)) / sample_rate * 1e3  # 轉換為毫秒
# amplitude = np.abs(signal)

# # 繪製完整信號的時域幅度
# plt.figure(figsize=(15, 8))
# plt.plot(time_axis, amplitude, 'b-', linewidth=0.5)
# plt.title('Signal Time Domain Amplitude')
# plt.xlabel('Time (ms)')
# plt.ylabel('Amplitude')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('signal_time_domain_amplitude.png', dpi=150, bbox_inches='tight')
# plt.show()
# plt.close()

# print("Signal time domain amplitude saved as 'signal_time_domain_amplitude.png'")

# # 對整個signal做FFT分析
# print("\n=== FFT Analysis of Complete Signal ===")

# # FFT分析
# fft_signal = np.fft.fft(signal)
# freq_axis = np.fft.fftfreq(len(signal), 1/sample_rate)
# freq_shifted = np.fft.fftshift(freq_axis)
# fft_shifted = np.fft.fftshift(fft_signal)

# # 計算功率譜密度
# power_spectrum = np.abs(fft_shifted)**2
# power_spectrum_db = 10 * np.log10(power_spectrum + 1e-12)

# # 繪製完整信號的FFT分析
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# fig.suptitle('Complete Signal FFT Analysis', fontsize=14)

# # 時域信號 - 幅度
# time_axis = np.arange(len(signal)) / sample_rate * 1e3  # 轉換為毫秒
# axes[0, 0].plot(time_axis, np.abs(signal), 'b-', linewidth=0.5)
# axes[0, 0].set_title('Time Domain - Signal Amplitude')
# axes[0, 0].set_xlabel('Time (ms)')
# axes[0, 0].set_ylabel('Amplitude')
# axes[0, 0].grid(True, alpha=0.3)

# # 頻域信號 - 幅度譜
# axes[0, 1].plot(freq_shifted/1e6, np.abs(fft_shifted), 'g-', linewidth=1)
# axes[0, 1].set_title('Frequency Domain - Magnitude Spectrum')
# axes[0, 1].set_xlabel('Frequency (MHz)')
# axes[0, 1].set_ylabel('Magnitude')
# axes[0, 1].grid(True, alpha=0.3)
# axes[0, 1].set_xlim(-sample_rate/2/1e6, sample_rate/2/1e6)  # 完整頻譜

# # 功率譜密度 (dB)
# axes[1, 0].plot(freq_shifted/1e6, power_spectrum_db, 'purple', linewidth=1)
# axes[1, 0].set_title('Power Spectral Density (dB)')
# axes[1, 0].set_xlabel('Frequency (MHz)')
# axes[1, 0].set_ylabel('Power (dB)')
# axes[1, 0].grid(True, alpha=0.3)
# axes[1, 0].set_xlim(-sample_rate/2/1e6, sample_rate/2/1e6)  # 完整頻譜

# # 相位譜
# phase_spectrum = np.angle(fft_shifted)
# axes[1, 1].plot(freq_shifted/1e6, phase_spectrum, 'orange', linewidth=1)
# axes[1, 1].set_title('Phase Spectrum')
# axes[1, 1].set_xlabel('Frequency (MHz)')
# axes[1, 1].set_ylabel('Phase (radians)')
# axes[1, 1].grid(True, alpha=0.3)
# axes[1, 1].set_xlim(-sample_rate/2/1e6, sample_rate/2/1e6)  # 完整頻譜

# plt.tight_layout()
# plt.savefig('complete_signal_fft_analysis.png', dpi=150, bbox_inches='tight')
# plt.show()
# plt.close()

# print("Complete signal FFT analysis saved as 'complete_signal_fft_analysis.png'")

# # 頻譜統計資訊
# print(f"\n=== FFT Statistics ===")
# max_power_idx = np.argmax(power_spectrum)
# peak_freq = freq_shifted[max_power_idx]
# print(f"Peak frequency: {peak_freq/1e6:.2f} MHz")
# print(f"Peak power: {np.max(power_spectrum_db):.2f} dB")
# print(f"Frequency resolution: {sample_rate/len(signal)/1e3:.2f} kHz")

# # 統計資訊
# print(f"\n=== Signal Amplitude Statistics ===")
# print(f"Signal length: {len(signal)} samples")
# print(f"Signal duration: {len(signal)/sample_rate:.3f} seconds")
# print(f"Max amplitude: {np.max(amplitude):.6f}")
# print(f"Min amplitude: {np.min(amplitude):.6f}")
# print(f"Mean amplitude: {np.mean(amplitude):.6f}")
# print(f"RMS amplitude: {np.sqrt(np.mean(amplitude**2)):.6f}")

# Convert to interleaved IQ data for HackRF (8-bit signed)
signal *= 127
iq_data = np.empty(2 * len(signal), dtype=np.int8)
iq_data[0::2] = signal.real.astype(np.int8)
iq_data[1::2] = signal.imag.astype(np.int8)

# Save to .iq file
iq_data.tofile('signal.iq')
# iq_data.tofile('fhss_signal.iq')

# print("\n=== Plotting IQ Signal for Each Hop ===")

# # 分析前10個hop
# for hop_idx in range(min(10, num_hops)):
#     start_idx = hop_idx * samples_per_hop * 2  # IQ data是交錯的，所以乘以2
#     end_idx = start_idx + samples_per_hop * 2
    
#     hop_iq = iq_data[start_idx:end_idx]
#     hop_i = hop_iq[0::2]
#     hop_q = hop_iq[1::2]
#     hop_complex = hop_i + 1j * hop_q
    
#     # 時域和頻域分析
#     hop_time = np.arange(len(hop_i)) / sample_rate * 1e6  # 轉換為微秒
    
#     # FFT分析
#     fft_hop = np.fft.fft(hop_complex)
#     freq_axis = np.fft.fftfreq(len(hop_complex), 1/sample_rate)
#     freq_shifted = np.fft.fftshift(freq_axis)
#     fft_shifted = np.fft.fftshift(fft_hop)
    
#     # 繪製時域和頻域圖
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     fig.suptitle(f'Hop {hop_idx} - IQ Signal (Center: {hop_sequence[hop_idx]/1e6:.1f} MHz)', fontsize=14)
    
#     # 時域圖
#     axes[0].plot(hop_time, hop_i, 'b-', linewidth=1, label='I')
#     axes[0].plot(hop_time, hop_q, 'r-', linewidth=1, label='Q')
#     axes[0].set_title('Time Domain - I and Q Components')
#     axes[0].set_xlabel('Time (μs)')
#     axes[0].set_ylabel('Amplitude (INT8)')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)
    
#     # 頻域圖
#     axes[1].plot(freq_shifted/1e6, np.abs(fft_shifted), 'g-', linewidth=1)
#     axes[1].set_title('Frequency Domain - IQ Spectrum')
#     axes[1].set_xlabel('Frequency (MHz)')
#     axes[1].set_ylabel('Magnitude')
#     axes[1].grid(True, alpha=0.3)
#     axes[1].set_xlim(-sample_rate/2/1e6, sample_rate/2/1e6)  # 完整頻譜
    
#     plt.tight_layout()
#     plt.savefig(f'iq_hop_{hop_idx}_analysis.png', dpi=150, bbox_inches='tight')
#     plt.show()
#     plt.close()
    
#     print(f"IQ Hop {hop_idx} analysis saved as 'iq_hop_{hop_idx}_analysis.png'")

# print("All IQ hop analyses completed!")

# # STFT分析
# print("\n=== STFT Analysis ===")
# from scipy import signal as scipy_signal

# # 使用原始複數信號進行STFT分析
# print("Performing STFT on the complex signal...")

# # STFT參數
# window_length = 20000  # 窗長
# overlap = window_length*0.3  # 重疊長度
# nperseg = window_length
# noverlap = overlap

# # 執行STFT
# f_stft, t_stft, Zxx = scipy_signal.stft(signal, fs=sample_rate, 
#                                        nperseg=nperseg, noverlap=noverlap)

# # 繪製STFT時頻圖
# plt.figure(figsize=(12, 8))

# # 時頻圖
# plt.pcolormesh(t_stft*1000, f_stft/1e6, np.abs(Zxx), shading='gouraud')
# plt.title('STFT - Time-Frequency Analysis')
# plt.xlabel('Time (ms)')
# plt.ylabel('Frequency (MHz)')
# plt.colorbar(label='Magnitude')
# plt.ylim(-sample_rate/2/1e6, sample_rate/2/1e6)  # 完整頻譜範圍

# plt.tight_layout()
# plt.savefig('stft_analysis_int8.png', dpi=150, bbox_inches='tight')
# plt.show()
# plt.close()

# print("STFT analysis saved as 'stft_analysis_int8.png'")

# # 額外：繪製STFT的3D視圖
# print("Generating 3D STFT visualization...")
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 創建網格
# T, F = np.meshgrid(t_stft*1000, f_stft/1e6)
# Z = np.abs(Zxx)

# # 3D表面圖
# surf = ax.plot_surface(T, F, Z, cmap='viridis', alpha=0.8)
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Frequency (MHz)')
# ax.set_zlabel('Magnitude')
# ax.set_title('3D STFT - Time-Frequency Analysis')

# plt.tight_layout()
# plt.savefig('stft_3d_analysis_int8.png', dpi=150, bbox_inches='tight')
# plt.show()
# plt.close()

# print("3D STFT analysis saved as 'stft_3d_analysis_int8.png'")

# Example HackRF transmit command (run separately):
# hackrf_transfer -t fhss_signal.iq -f 2400000000 -s 20000000 -a 1
