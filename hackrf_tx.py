import subprocess

class HackRFOne():
    def __init__(self):
        self.serial_number = "000000000000000c66c63dc2d328b83"
        self.file_name = None
        self.center_freq = None
        self.sample_rate = 20000000
        self.amplifier_enable = 1
        self.antenna_port_enable = 1

        self.tx_vga_gain = 30 #TX VGA gain, 0-47 dB,step = 1 dB

    
    def TX(self):

        cmd = f"hackrf_transfer -t {self.file_name} -f {self.center_freq} -s {self.sample_rate} -a {self.amplifier_enable} -R -p {self.antenna_port_enable} -x {self.tx_vga_gain}"
        print(cmd)
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    hackrf = HackRFOne()
    hackrf.TX()
