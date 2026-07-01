# 氨基甲酸铵分解平衡常数测定控制程序的一部分，用于连接硬件
# 项目作者：李峙德，刘一弘
# 文件作者：李峙德
# 邮箱：contact@chemview.net
# 最后更新：2026-06-26
# Hardware connection helpers for the ammonium carbamate decomposition equilibrium constant experiment
# Project authors Li Zhide and Liu Yihong
# File author Li Zhide
# Email contact@chemview.net
# Last updated 2026-06-26
import cv2
import serial
import time

class Cap:

    def __init__(self, cap_num: int) -> None:
        self.cap_num = cap_num
        self.cap = None
        self.frame = None

    def open_cap(self) -> None:
        self.cap = cv2.VideoCapture(int(self.cap_num))
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0) # Enable auto exposure
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0) # Turn off auto white balance

        if not self.cap.isOpened():
            raise Exception('CapConnectionError')

    def close_cap(self) -> None:
        self.cap.release()

    def get_frame(self) -> None:
        if not self.cap:
            self.open_cap()

        ret, frame = self.cap.read()
        if not ret:
            raise Exception('CapReadError')

        self.frame = frame

# Serial control
class SerialPort:

    def __init__(self, com: str, baud: int =9600, timeout: int =3) -> None:
        self.com = com
        self.baud = baud
        self.timeout = timeout
        self.serial_port = serial.Serial(com, baud, timeout=timeout)  # Keep a read timeout
        time.sleep(2) # Give the device time to connect

    def send(self, command: str) -> None:
        if not command.endswith('\n'):
            command += '\n'
        self.serial_port.write(command.encode())
        time.sleep(0.1)
    
    def read(self) -> str:
        try:
            response_bytes = self.serial_port.readline()
            return response_bytes
        except:
            return None
    
    def readpre(self):
        data_bytes = None
        try:
            response_bytes = self.serial_port.read(7)  # Seven bytes per frame
            if len(response_bytes) == 7:
                data_bytes = response_bytes
        except:
            return None
        
        if data_bytes is None or len(data_bytes) != 7:
            return None
    
        current_high = data_bytes[3]  # 0x0B
        current_low = data_bytes[4]   # 0xB8
        
        current_raw = (current_high << 8) | current_low # Combine into a 16 bit value
        current = current_raw * 10000 / 249 # Keep the pressure math stable
        pressure = (current * 125 - 14962000) / 100000
        return [pressure, current/10000]

    def close(self):
        self.serial_port.close()

# Relay control
class Port:

    def __init__(self, port: str) -> None:
        self.serial_port = SerialPort(port)

    def open(self, num: int) -> None:
        self.serial_port.send(f'OPEN {num}')

    def close(self, num: int) -> None:
        self.serial_port.send(f'CLOSE {num}')

    def release_port(self) -> None:
        self.serial_port.close()

    def release(self) -> None:
        for i in range(4):
            self.close(i+1)
