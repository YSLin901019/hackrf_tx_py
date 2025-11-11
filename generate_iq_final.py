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
        self.num_sinc_waves = 300

        self.start_frequency = None
        self.stop_frequency = None
        self.center_frequency = None
        # self.hopping_frequency_list = np.array([-7.5e6,-4e6, -1.5e6, 1.5e6, 4e6, 7.5e6])
        self.hopping_frequency_list = np.array([-4e6, -1.5e6, 1.5e6, 4e6, 7.5e6])
        self.single_frequency_list = np.array([0])
        self.signal_mode = None # will be "hopping" or "single"

        self.hackrf_one = HackRFOne()

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
        self.start_frequency = float(config['start_frequency']) * 1e6  # MHz 轉 Hz
        self.stop_frequency = float(config['stop_frequency']) * 1e6  # MHz 轉 Hz
        self.bandwidth = float(config['bandwidth'])  # 確保是數值類型
        self.tx_vga_gain = int(config['tx_vga_gain'])  # dBm
        self.cw_amplitude = int(config['cw_amplitude'])
        self.hop_rate = float(config['hop_rate'])
        self.hop_duration = 1.0 / self.hop_rate  # 0.001 seconds per hop
        self.num_hops = int(self.total_duration * self.hop_rate)
        self.samples_per_hop = int(self.sample_rate * self.hop_duration)

        # 計算 power 對應的 tx_vga_gain
        tx_vga_gain = self.tx_vga_gain

        # 配置到 hackrf_one
        self.hackrf_one.file_name = self.file_name
        self.hackrf_one.tx_vga_gain = tx_vga_gain

    def generate_hopping_iq(self):
        self.hackrf_one.center_freq = (self.start_frequency + self.stop_frequency) / 2
        # 獲取到 start_frequency 和 stop_frequency 後重新設計 hopping_frequency_list
        
        # Generate a pseudo-random hop sequence
        np.random.seed(42)  # For reproducibility
        hop_sequence = np.random.choice(self.hopping_frequency_list, size=self.num_hops, replace=True)

        # Time vector for one hop
        t_hop = np.linspace(0, self.hop_duration, self.samples_per_hop, endpoint=False)

        # Generate the complex baseband signal with single sinc wave per hop
        signal = np.zeros(self.num_hops * self.samples_per_hop, dtype=np.complex64)
        for i in range(self.num_hops):
            f_center = hop_sequence[i]
            start_idx = i * self.samples_per_hop
            end_idx = start_idx + self.samples_per_hop
            
            # 生成單個sinc波 - 時域上就是一個sinc函數
            sinc_signal = np.zeros_like(t_hop, dtype=np.complex64)
            
            # 在hop時間內均勻分佈多個sinc波
            for k in range(self.num_sinc_waves):
                # 計算每個sinc波的時間偏移
                offset = (k - self.num_sinc_waves//2) * self.hop_duration / self.num_sinc_waves
                t_offset = t_hop - self.hop_duration/2 + offset
                
                # 生成sinc波
                sinc_component = np.sinc(self.bandwidth * t_offset)
                
                # 加入固定幅度和相位變化
                amplitude = 1.0  # 固定振幅
                phase = np.random.uniform(0, 2*np.pi)
                
                # 組合多個sinc波
                sinc_signal += amplitude * sinc_component * np.exp(1j * phase)
            
            # 加入載波頻率
            carrier = 1*np.exp(1j * 2 * np.pi * f_center * t_hop)
            hop_signal = sinc_signal * carrier
            
            signal[start_idx:end_idx] = hop_signal * np.hanning(self.samples_per_hop)

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
        # Generate a pseudo-random hop sequence
        np.random.seed(42)  # For reproducibility
        hop_sequence = np.random.choice(self.single_frequency_list, size=self.num_hops, replace=True)
        # hop_sequence = np.tile(np.array([-7.5e6, -2.5e6, 2.5e6, 7.5e6]), 10)

        # Time vector for one hop
        t_hop = np.linspace(0, self.hop_duration, self.samples_per_hop, endpoint=False)

        # Generate the complex baseband signal with single sinc wave per hop
        signal = np.zeros(self.num_hops * self.samples_per_hop, dtype=np.complex64)
        for i in range(self.num_hops):
            f_center = hop_sequence[i]
            start_idx = i * self.samples_per_hop
            end_idx = start_idx + self.samples_per_hop
            
            # 生成單個sinc波 - 時域上就是一個sinc函數
            sinc_signal = np.zeros_like(t_hop, dtype=np.complex64)
            
            # 在hop時間內均勻分佈多個sinc波
            for k in range(self.num_sinc_waves):
                # 計算每個sinc波的時間偏移
                offset = (k - self.num_sinc_waves//2) * self.hop_duration / self.num_sinc_waves
                t_offset = t_hop - self.hop_duration/2 + offset
                
                # 生成sinc波
                sinc_component = np.sinc(self.bandwidth * t_offset)
                
                # 加入固定幅度和相位變化
                amplitude = 1.0  # 固定振幅
                phase = np.random.uniform(0, 2*np.pi)
                
                # 組合多個sinc波
                sinc_signal += amplitude * sinc_component * np.exp(1j * phase)
            
            # 加入載波頻率
            carrier = 1*np.exp(1j * 2 * np.pi * f_center * t_hop)
            hop_signal = sinc_signal * carrier
            
            signal[start_idx:end_idx] = hop_signal * np.hanning(self.samples_per_hop)

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
