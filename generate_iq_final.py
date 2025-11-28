import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from hackrf_tx import HackRFOne
import yaml

class IQGenerator():
    def __init__(self):
        self.file_name = None
        self.sample_rate = 20e6  
        self.hop_rate = None  
        self.hop_duration = None  
        self.num_hops = None
        self.samples_per_hop = None
        self.total_duration = 0.1  

        self.bandwidth = None  
        self.duty_cycle = 0.5  # 預設 50% duty cycle

        self.center_frequency = None
        # self.hopping_frequency_list = np.array([-7.5e6,-4e6, -1.5e6, 1.5e6, 4e6, 7.5e6])
        self.hopping_frequency_list = np.array([-2e6, -1e6, 0, 1e6, 2e6])
        self.single_frequency_list = np.array([0])
        self.signal_mode = None # will be "hopping" or "single"
        self.num_carriers = 300  # 使用多個載波來填滿頻寬
        self.antenna_index = None
        self.antenna_angle = None

        self.hackrf_one = HackRFOne()
        
        # HackRF 輸出功率實測數據表
        # 格式: (freq_MHz, tx_vga_gain, measured_power_dBm)
        self.tx_gain_table = [
            # tx_vga_gain = 10
            (1350, 10, -78.03),
            (1400, 10, -78.37),
            (1500, 10, -78.59),
            (1600, 10, -78.16),
            (1700, 10, -80.32),
            (1800, 10, -83.76),
            (1900, 10, -84.06),
            (2000, 10, -81.02),
            (2100, 10, -83.01),
            (2170, 10, -75.46),
            (2175, 10, -75.77),
            (2180, 10, -75.65),
            (2200, 10, -74.91),
            (2300, 10, -70.93),
            (2400, 10, -69.62),
            (2500, 10, -71.05),
            (2600, 10, -73.03),
            (2700, 10, -71.33),
            # tx_vga_gain = 15
            (1350, 15, -74.03),
            (1400, 15, -74.08),
            (1500, 15, -74.49),
            (1600, 15, -73.81),
            (1700, 15, -76.12),
            (1800, 15, -79.89),
            (1900, 15, -80.03),
            (2000, 15, -76.71),
            (2100, 15, -78.44),
            (2170, 15, -70.77),
            (2175, 15, -71.11),
            (2200, 15, -70.49),
            (2300, 15, -66.37),
            (2400, 15, -65.10),
            (2500, 15, -66.43),
            (2600, 15, -68.59),
            (2700, 15, -66.89),
            # tx_vga_gain = 20
            (1350, 20, -67.48),
            (1400, 20, -67.8),
            (1500, 20, -68.34),
            (1600, 20, -67.53),
            (1700, 20, -69.86),
            (1800, 20, -73.68),
            (1900, 20, -73.83),
            (2000, 20, -70.58),
            (2100, 20, -72.86),
            (2170, 20, -65.03),
            (2175, 20, -65.43),
            (2200, 20, -64.53),
            (2300, 20, -60.43),
            (2400, 20, -58.73),
            (2500, 20, -60.24),
            (2600, 20, -62.08),
            (2700, 20, -60.08),
        ]

    def get_measured_power(self, freq_MHz: float, tx_vga_gain: int) -> float:
        """
        根據輸入頻率與 tx_vga_gain 查表，回傳對應功率(dBm)。
        若無精確頻點，則使用線性插值計算。
        注意：
        1. 2170 MHz 是硬體分界點，不會跨越此界限進行插值。
        2. 查詢表基於 5 MHz 頻寬測量，會根據實際頻寬自動修正功率值。
        """
        same_gain_entries = [e for e in self.tx_gain_table if e[1] == tx_vga_gain]
        if not same_gain_entries:
            print(f"⚠️ 無對應 tx_vga_gain={tx_vga_gain} 的資料")
            return None

        # 硬體分界點
        HARDWARE_BOUNDARY = 2170.0
        
        # 根據硬體分界，分成兩組數據
        if freq_MHz <= HARDWARE_BOUNDARY:
            # 使用 <= 2170 MHz 的硬體數據
            valid_entries = [e for e in same_gain_entries if e[0] <= HARDWARE_BOUNDARY]
        else:
            # 使用 > 2170 MHz 的硬體數據
            valid_entries = [e for e in same_gain_entries if e[0] > HARDWARE_BOUNDARY]
        
        if not valid_entries:
            print(f"⚠️ 在對應硬體區間內無 tx_vga_gain={tx_vga_gain} 的資料")
            return None
        
        # 檢查是否有精確匹配
        exact_match = [e for e in valid_entries if abs(e[0] - freq_MHz) < 0.01]
        if exact_match:
            base_power = exact_match[0][2]
        else:
            # 按頻率排序
            valid_entries = sorted(valid_entries, key=lambda x: x[0])
            
            # 如果查詢頻率在範圍外，返回最近的端點值
            if freq_MHz < valid_entries[0][0]:
                base_power = valid_entries[0][2]
            elif freq_MHz > valid_entries[-1][0]:
                base_power = valid_entries[-1][2]
            else:
                # 線性插值：找到相鄰的兩個點
                base_power = None
                for i in range(len(valid_entries) - 1):
                    f1, gain1, p1 = valid_entries[i]
                    f2, gain2, p2 = valid_entries[i + 1]
                    
                    if f1 <= freq_MHz <= f2:
                        # 線性插值公式：p = p1 + (p2 - p1) * (f - f1) / (f2 - f1)
                        base_power = p1 + (p2 - p1) * (freq_MHz - f1) / (f2 - f1)
                        break
                
                # 如果沒有找到合適的區間（理論上不應該發生），返回最近的點
                if base_power is None:
                    nearest = min(valid_entries, key=lambda x: abs(x[0] - freq_MHz))
                    base_power = nearest[2]
        
        # 頻寬修正：查詢表基於 5 MHz 測量
        # 修正公式：correction_dB = 10 * log10(5 MHz / 實際頻寬)
        REFERENCE_BANDWIDTH = 5e6  # 5 MHz
        bandwidth_correction = 10 * np.log10(REFERENCE_BANDWIDTH / self.bandwidth)
        
        # 應用修正
        corrected_power = base_power + bandwidth_correction 

        # 直接調整以利優話
        
        corrected_power += 0

        if self.antenna_index == 2:
            corrected_power -= 2.5
        elif self.antenna_index == 4:
            corrected_power += 1.5

        return corrected_power

    def generate_iq(self):
        if self.signal_mode == "hopping":
            self.generate_hopping_iq()
        elif self.signal_mode == "single":
            self.generate_single_iq()
        elif self.signal_mode == "cw":
            self.generate_cw()
        else:
            raise ValueError("Invalid signal mode")

    def load_config(self):
        # 讀取 tx_signal_setting.yaml 檔案
        with open('tx_signal_setting.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # 讀取並轉換參數類型
        self.signal_mode = config['signal_mode']
        self.center_frequency = float(config['center_frequency']) * 1e6  # MHz 轉 Hz
        self.bandwidth = float(config['bandwidth'])  # 確保是數值類型
        self.tx_vga_gain = int(config['tx_vga_gain'])  # dBm
        self.cw_amplitude = int(config['cw_amplitude'])
        self.hop_rate = float(config['hop_rate'])
        self.hop_duration = 1.0 / self.hop_rate  # 0.001 seconds per hop
        self.num_hops = int(self.total_duration * self.hop_rate)
        self.samples_per_hop = int(self.sample_rate * self.hop_duration)
        self.antenna_index = int(config['antenna_index'])
        self.antenna_angle = float(22.5+(self.antenna_index-1)*45)
        
        # 讀取 duty_cycle，如果沒有設定則使用預設值
        self.duty_cycle = float(config.get('duty_cycle', 0.5))  # 預設 50%

        # 計算 power 對應的 tx_vga_gain
        tx_vga_gain = self.tx_vga_gain

        # 配置到 hackrf_one
        self.hackrf_one.file_name = self.file_name
        self.hackrf_one.tx_vga_gain = tx_vga_gain

    def generate_hopping_iq(self):
        # 使用固定的中心頻率，在 20MHz 頻寬內跳頻
        self.hackrf_one.center_freq = self.center_frequency
        
        # 打印實際發射頻率資訊
        print("=" * 60)
        print("跳頻訊號生成資訊")
        print("=" * 60)
        print(f"中心頻率 (fc): {self.center_frequency / 1e6:.2f} MHz")
        print(f"跳頻偏移量: {[f'{f/1e6:.2f} MHz' for f in self.hopping_frequency_list]}")
        print(f"\n實際發射頻率 (fc + 偏移量):")
        for index, offset in enumerate(self.hopping_frequency_list):
            actual_freq = self.center_frequency + offset
            print(f"Channel {index+1}: {actual_freq / 1e6:.2f} MHz (中心頻 {self.center_frequency/1e6:.2f} MHz + 偏移 {offset/1e6:.2f} MHz)")
        
        # 計算並顯示輸出功率（含頻寬修正）
        measured_power = self.get_measured_power(self.center_frequency / 1e6, self.tx_vga_gain)
        bandwidth_correction = 10 * np.log10(5e6 / self.bandwidth) if self.bandwidth != 5e6 else 0.0
        print(f"\n訊號頻寬: {self.bandwidth / 1e6:.2f} MHz")
        if abs(bandwidth_correction) > 0.01:
            print(f"輸出功率: {measured_power:.2f} dBm")
        else:
            print(f"輸出功率: {measured_power:.2f} dBm")
        print(f"跳頻速率: {self.hop_rate} Hz ({1000/self.hop_rate:.2f} ms/hop)")
        print(f"Duty Cycle: {self.duty_cycle * 100:.1f}%")
        print(f"訊號來源方向： {self.antenna_angle} 度")
        print("=" * 60 + "\n")
        
        # Generate a pseudo-random hop sequence
        # hopping_frequency_list 中的頻率是相對於中心頻率的偏移量
        np.random.seed(42)  # For reproducibility
        hop_sequence = np.random.choice(self.hopping_frequency_list, size=self.num_hops, replace=True)

        # Time vector for one hop
        t_hop = np.linspace(0, self.hop_duration, self.samples_per_hop, endpoint=False)

        # Generate the complex baseband signal with rectangular spectrum (square in frequency domain)
        signal = np.zeros(self.num_hops * self.samples_per_hop, dtype=np.complex64)
        
        # 計算 duty cycle 相關的樣本數
        active_samples = int(self.samples_per_hop * self.duty_cycle)  # 有訊號的樣本數
        
        # 設定切換時間（佔 active 時間的比例）
        transition_ratio = 0.02  # 2% 用於切換
        transition_samples = int(active_samples * transition_ratio)
        
        for i in range(self.num_hops):
            f_center = hop_sequence[i]
            start_idx = i * self.samples_per_hop
            end_idx = start_idx + self.samples_per_hop
            
            # 方法：使用多載波疊加來模擬矩形頻譜
            # 在頻域上產生矩形（佔據指定頻寬），時域上會是 sinc-like 的包絡
            hop_signal = np.zeros(self.samples_per_hop, dtype=np.complex64)
            
            # 只在 duty cycle 時間內生成訊號
            t_active = t_hop[:active_samples]
            active_signal = np.zeros(active_samples, dtype=np.complex64)
            
            # 在指定頻寬內均勻分布多個載波
            # 確保不超過 Nyquist 頻率限制
            nyquist_freq = self.sample_rate / 2  # ±10 MHz
            valid_carriers = 0  # 計數有效的載波數量
            
            for k in range(self.num_carriers):
                # 計算每個載波相對於中心頻率的偏移
                freq_offset = (k - self.num_carriers/2) * (self.bandwidth / self.num_carriers)
                carrier_freq = f_center + freq_offset
                
                # 檢查是否超出 Nyquist 頻率範圍，如果超出則跳過此載波
                if abs(carrier_freq) >= nyquist_freq:
                    continue
                
                valid_carriers += 1
                
                # 生成載波（只在 active 時間內）
                carrier = np.exp(1j * 2 * np.pi * carrier_freq * t_active)
                
                # 隨機相位避免峰值過高
                phase = np.random.uniform(0, 2*np.pi)
                active_signal += np.exp(1j * phase) * carrier
            
            # 正規化振幅（使用實際有效的載波數量）
            if valid_carriers > 0:
                active_signal /= np.sqrt(valid_carriers)
            
            # 應用窗函數（只在 active 部分）
            window = np.ones(active_samples, dtype=np.float32)
            if transition_samples > 0:
                # 上升邊緣
                window[:transition_samples] = np.linspace(0, 1, transition_samples) ** 2
                # 下降邊緣
                window[-transition_samples:] = np.linspace(1, 0, transition_samples) ** 2
            
            # 將 active 訊號放入 hop_signal 的前面部分，後面保持為 0
            hop_signal[:active_samples] = active_signal * window
            
            signal[start_idx:end_idx] = hop_signal

        # Normalize the signal
        signal /= np.max(np.abs(signal))

        # Convert to interleaved IQ data for HackRF (8-bit signed)
        signal *= 127
        iq_data = np.empty(2 * len(signal), dtype=np.int8)
        iq_data[0::2] = signal.real.astype(np.int8)
        iq_data[1::2] = signal.imag.astype(np.int8)

        # Save to .iq file
        iq_data.tofile('hopping_signal.iq')
        self.file_name = './hopping_signal.iq'

    def generate_single_iq(self):
        self.hackrf_one.center_freq = self.center_frequency
        
        # 打印實際發射頻率資訊
        print("=" * 60)
        print("定頻訊號生成資訊")
        print("=" * 60)
        print(f"中心頻率 (fc): {self.center_frequency / 1e6:.2f} MHz")
        print(f"\n實際發射頻率:")
        for offset in self.single_frequency_list:
            actual_freq = self.center_frequency + offset
            print(f"  {actual_freq / 1e6:.2f} MHz (固定頻率，無跳頻)")
        
        # 計算並顯示輸出功率（含頻寬修正）
        measured_power = self.get_measured_power(self.center_frequency / 1e6, self.tx_vga_gain)
        bandwidth_correction = 10 * np.log10(5e6 / self.bandwidth) if self.bandwidth != 5e6 else 0.0
        print(f"\n訊號頻寬: {self.bandwidth / 1e6:.2f} MHz")
        if abs(bandwidth_correction) > 0.01:
            print(f"輸出功率: {measured_power:.2f} dBm")
        else:
            print(f"輸出功率: {measured_power:.2f} dBm")
        print(f"開關速率: {self.hop_rate} Hz ({1000/self.hop_rate:.2f} ms/hop)")
        print(f"Duty Cycle: {self.duty_cycle * 100:.1f}%")
        print(f"訊號來源方向： {self.antenna_angle} 度")
        print("=" * 60 + "\n")
        
        # Generate a pseudo-random hop sequence
        np.random.seed(42)  # For reproducibility
        hop_sequence = np.random.choice(self.single_frequency_list, size=self.num_hops, replace=True)
        # hop_sequence = np.tile(np.array([-7.5e6, -2.5e6, 2.5e6, 7.5e6]), 10)

        # Time vector for one hop
        t_hop = np.linspace(0, self.hop_duration, self.samples_per_hop, endpoint=False)

        # Generate the complex baseband signal with rectangular spectrum (square in frequency domain)
        signal = np.zeros(self.num_hops * self.samples_per_hop, dtype=np.complex64)
        
        # 計算 duty cycle 相關的樣本數
        active_samples = int(self.samples_per_hop * self.duty_cycle)  # 有訊號的樣本數
        
        # 設定切換時間（佔 active 時間的比例）
        transition_ratio = 0.02  # 2% 用於切換
        transition_samples = int(active_samples * transition_ratio)
        
        for i in range(self.num_hops):
            f_center = hop_sequence[i]
            start_idx = i * self.samples_per_hop
            end_idx = start_idx + self.samples_per_hop
            
            # 方法：使用多載波疊加來模擬矩形頻譜
            # 在頻域上產生矩形（佔據指定頻寬），時域上會是 sinc-like 的包絡
            hop_signal = np.zeros(self.samples_per_hop, dtype=np.complex64)
            
            # 只在 duty cycle 時間內生成訊號
            t_active = t_hop[:active_samples]
            active_signal = np.zeros(active_samples, dtype=np.complex64)
            
            # 在指定頻寬內均勻分布多個載波
            # 確保不超過 Nyquist 頻率限制
            nyquist_freq = self.sample_rate / 2  # ±10 MHz
            valid_carriers = 0  # 計數有效的載波數量
            
            for k in range(self.num_carriers):
                # 計算每個載波相對於中心頻率的偏移
                freq_offset = (k - self.num_carriers/2) * (self.bandwidth / self.num_carriers)
                carrier_freq = f_center + freq_offset
                
                # 檢查是否超出 Nyquist 頻率範圍，如果超出則跳過此載波
                if abs(carrier_freq) >= nyquist_freq:
                    continue
                
                valid_carriers += 1
                
                # 生成載波（只在 active 時間內）
                carrier = np.exp(1j * 2 * np.pi * carrier_freq * t_active)
                
                # 隨機相位避免峰值過高
                phase = np.random.uniform(0, 2*np.pi)
                active_signal += np.exp(1j * phase) * carrier
            
            # 正規化振幅（使用實際有效的載波數量）
            if valid_carriers > 0:
                active_signal /= np.sqrt(valid_carriers)
            
            # 應用窗函數（只在 active 部分）
            window = np.ones(active_samples, dtype=np.float32)
            if transition_samples > 0:
                # 上升邊緣
                window[:transition_samples] = np.linspace(0, 1, transition_samples) ** 2
                # 下降邊緣
                window[-transition_samples:] = np.linspace(1, 0, transition_samples) ** 2
            
            # 將 active 訊號放入 hop_signal 的前面部分，後面保持為 0
            hop_signal[:active_samples] = active_signal * window
            
            signal[start_idx:end_idx] = hop_signal

        # Normalize the signal
        signal /= np.max(np.abs(signal))

        # Convert to interleaved IQ data for HackRF (8-bit signed)
        signal *= 127
        iq_data = np.empty(2 * len(signal), dtype=np.int8)
        iq_data[0::2] = signal.real.astype(np.int8)
        iq_data[1::2] = signal.imag.astype(np.int8)

        # Save to .iq file
        iq_data.tofile('single_signal.iq')
        self.file_name = './single_signal.iq'

    def generate_cw(self):

        self.hackrf_one.center_freq = self.center_frequency
        self.hackrf_one.cw_signal_amplitude = self.cw_amplitude

        
    def tx_signal(self):
        if self.signal_mode == "hopping":
            self.hackrf_one.file_name = "hopping_signal.iq"
            self.hackrf_one.TX()
        elif self.signal_mode == "single":
            self.hackrf_one.file_name = "single_signal.iq"
            self.hackrf_one.TX()
        elif self.signal_mode == "cw":
            self.hackrf_one.CW()
        else:
            raise ValueError("Invalid signal mode")


if __name__ == "__main__":
    iq_generator = IQGenerator()
    iq_generator.load_config()
    iq_generator.generate_iq()
    iq_generator.tx_signal()
