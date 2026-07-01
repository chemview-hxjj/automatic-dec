# 氨基甲酸铵分解平衡常数测定控制程序的一部分，用于实现判断与控制逻辑
# 项目作者：李峙德，刘一弘
# 文件作者：李峙德
# 邮箱：contact@chemview.net
# 最后更新：2026-06-26
# Detector and control flow for the ammonium carbamate decomposition equilibrium constant experiment
# Project authors Li Zhide and Liu Yihong
# File author Li Zhide
# Email contact@chemview.net
# Last updated 2026-06-26
import ast
import time
from datetime import datetime
import threading
import visual_feedback
import hw_connect
class Timer:

    def __init__(self) -> None:
        self.started = False
        self.init_time = 0
        self.curr_time = 0
        self.elap_time = 0

    def start(self) -> None:
        if not self.started:
            self.init_time = time.time()
            self.started = True

    def restart(self) -> None:
        self.init_time = time.time()

    def get_elap_time(self) -> float:
        self.curr_time = time.time()
        self.elap_time = self.curr_time - self.init_time
        return float(self.elap_time)
    
    def stop(self) -> None:
        self.started = False
        # self.init_time = 0
        # self.curr_time = 0
        # self.elap_time = 0

class Detector:

    def __init__(self) -> None:
        self.utubedet = None
        self.valves = None
        self.pregauge = None
        self.cap = None
        self.mp = None
        self.cap_num = 0
        self.port = None
        self.gaup = None
        self.status = 0
        self.last_status = 10
        self.pressure = 0
        self.delay_large = 0.5
        self.hsv_lower1 = None
        self.hsv_upper1 = None
        self.hsv_lower2 = None
        self.hsv_upper2 = None
        self.large_diff_threshold = 50
        self.small_diff_threshold = 5
        self.bottom_threshold = 0.85
        self.running = False
        self.airin = False
        self.vacuum_threshold = -90
        self.vacuum_hold_time = 600
        self.vacuum_cache_running = False
        self.vacuum_reached_since = None
        self.ispreview = True
        self.usemask = False
        self.timer = Timer()
        self.elapsed_time = 30
        self.right_level = 0
        self.left_level = 0
        self.diff = 0
        self.pre_current = None
        self.t = None
        self.last_t = None

    
    def get_cvt_status(self) -> int: # Bridge status values from visual_feedback 0 unknown 1 left high large 2 left high small 3 right high large 4 right high small 7 balanced
        self.utubedet.get_liquid_state()
        status = self.utubedet.current_state.value
        raw_status = self.utubedet.result['raw_state_code']
        self.right_level = self.utubedet.result['right_level']
        self.left_level = self.utubedet.result['left_level']
        self.diff = self.utubedet.result['height_difference']
        self.status = status
        self.usm_status = raw_status
        self.usm_diff = self.utubedet.result['usm_height_difference']
        self.usm_right_level = self.utubedet.result['usm_right_level']
        self.usm_left_level = self.utubedet.result['usm_left_level']
        # t=time.strftime('%H:%M:%S', time.localtime()) # Time rounded to seconds
        self.t = datetime.now().strftime('%H:%M:%S.%f')[:-5]
        if self.t != self.last_t and self.running:
            self.mp.data(f'{self.t}, {self.pressure:.4f}, {self.status}, {self.pre_current:.4f}, {self.diff}, {self.right_level}, {self.left_level}, {self.usm_status}, {self.usm_diff}, {self.usm_right_level}, {self.usm_left_level}')
        self.last_t = self.t
        if self.last_status != self.status:
            self.last_status = self.status
            self.mp.log('sc', str(self.status))
        return status
    
    def _run_con(self) -> None:
        try:
            self.mp.send('wa')
            if not self.port:
                raise ValueError('Valve port is not configured')
            if not self.gaup:
                raise ValueError('Pressure gauge port is not configured')
            self.utubedet=visual_feedback.UTubeDetector(ast.literal_eval(self.hsv_lower1),
                ast.literal_eval(self.hsv_upper1),
                ast.literal_eval(self.hsv_lower2),
                ast.literal_eval(self.hsv_upper2),
                self.large_diff_threshold,
                self.small_diff_threshold,
                self.small_diff_threshold,
                self.bottom_threshold)
            self.valves=hw_connect.Port(self.port)
            self.pregauge=hw_connect.Port(self.gaup)
            self.cap=hw_connect.Cap(self.cap_num)
            self.cap.get_frame()
            self.utubedet.frame=self.cap.frame
            self.mp.send('re')
            self.preview()
            self.get_pressure()
        except Exception as e:
            self.mp.send('ce', f'{e}')

    def con(self) -> None:
        con_thread = threading.Thread(target=self._run_con)
        con_thread.daemon = True
        con_thread.start()

    def _run_get_pressure(self) -> None:
        while True:
            try:
                pressure_data = self.pregauge.serial_port.readpre()
                if pressure_data:
                    self.pressure, self.pre_current = pressure_data
            except Exception:
                pass

    def get_pressure(self) -> None:
        pres_thread = threading.Thread(target=self._run_get_pressure)
        pres_thread.daemon = True
        pres_thread.start()

    def _run_preview(self) -> None:
        try:
            while self.ispreview:
                self.cap.get_frame()
                self.utubedet.frame=self.cap.frame
                self.get_cvt_status()
        except Exception as e:
                self.mp.log('pe', f'{e}')
        
    def preview(self):
        self._preview_thread = threading.Thread(target=self._run_preview)
        self._preview_thread.daemon = True
        self._preview_thread.start()

    def _run_go_vacuum(self):
        try:
            self.mp.log('vs')
            self.vacuum_cache_running = True
            self.vacuum_reached_since = None
            self.valves.open(5)
            while self.vacuum_cache_running:
                if self.pressure <= self.vacuum_threshold:
                    if self.vacuum_reached_since is None:
                        self.vacuum_reached_since = time.time()
                    elif time.time() - self.vacuum_reached_since >= self.vacuum_hold_time:
                        self.stop_vacuum()
                        break
                else:
                    self.vacuum_reached_since = None
                time.sleep(1)

        except Exception as e:
            self.vacuum_cache_running = False
            self.mp.send('ae', f'{e}')

    def go_vacuum(self):
        if self.vacuum_cache_running:
            return
        self._go_vacuum_thread = threading.Thread(target=self._run_go_vacuum)
        self._go_vacuum_thread.daemon = True
        self._go_vacuum_thread.start()

    def _run_go_vacuum_2(self):
        try:
            self.mp.log('vs')
            self.valves.open(1)
            self.valves.open(2)
                    
        except Exception as e:
            self.mp.send('ae', f'{e}')

    def go_vacuum_2(self):
        self._go_vacuum_2_thread = threading.Thread(target=self._run_go_vacuum_2)
        self._go_vacuum_2_thread.daemon = True
        self._go_vacuum_2_thread.start()

    def _run_go_vacuum_3(self):
        try:
            self.mp.log('vs')
            self.valves.open(1)
            time.sleep(0.2)
            self.valves.close(1)
            time.sleep(0.3)
            self.valves.open(2)
            time.sleep(0.2)
            self.valves.close(2)
                    
        except Exception as e:
            self.mp.send('ae', f'{e}')

    def go_vacuum_3(self):
        self._go_vacuum_3_thread = threading.Thread(target=self._run_go_vacuum_3)
        self._go_vacuum_3_thread.daemon = True
        self._go_vacuum_3_thread.start()

    def _run_airin(self):
        try:
            self.mp.log('vs')
            self.airin = True
            while self.airin:
                self.valves.open(3)
                time.sleep(0.2)
                self.valves.close(3)
                time.sleep(0.3)
                self.valves.open(4)
                time.sleep(0.2)
                self.valves.close(4)
                self.airin = False
                    
        except Exception as e:
            self.mp.send('ae', f'{e}')

    def go_airin(self):
        self._run_airin_thread = threading.Thread(target=self._run_airin)
        self._run_airin_thread.daemon = True
        self._run_airin_thread.start()

    def _run_detect(self) -> None:
        try:
            self.pre_lim = -60 if self.pressure <= -60 else -40
            self.ptd_dict = {-60: {'lhl': [0.1, 0.5], 'lhs': [0.2, 1.25], 'rhl': [0.1, 0.5], 'rhs': [0.2, 1.25], 'rtl': [0.1, 0.1], 'ltl': [0.1, 0.1]}, 
                             -40: {'lhl': [0.1, 0.5], 'lhs': [0.2, 1.25], 'rhl': [0.1, 0.5], 'rhs': [0.2, 1.25], 'rtl': [0.1, 0.1], 'ltl': [0.1, 0.1]}}
            self.running = True
            '''
            RNF GV 8
            RTL GV 5
            LH L GV 1
            LH S GV 2
            RH S GA 4
            RH L GA 3
            LTL GA 6
            LNF GA 9
            '''
            while self.running:
                if self.status == 1: # Left high large
                    self.timer.restart()
                    self.valves.open(2)
                    self.valves.close(2)
                    time.sleep(self.ptd_dict[self.pre_lim]['lhl'][0])
                    self.valves.open(1)
                    self.valves.close(1)
                    time.sleep(self.ptd_dict[self.pre_lim]['lhl'][1])
                    
                elif self.status == 2: # Left high small
                    self.timer.restart()
                    self.valves.open(2)
                    self.valves.close(2)
                    time.sleep(self.ptd_dict[self.pre_lim]['lhs'][0])
                    self.valves.open(1)
                    self.valves.close(1)
                    time.sleep(self.ptd_dict[self.pre_lim]['lhs'][1])
                
                elif self.status == 3: # Right high large
                    self.timer.restart()
                    self.valves.open(3)
                    self.valves.close(3)
                    time.sleep(self.ptd_dict[self.pre_lim]['rhl'][0])
                    self.valves.open(4)
                    self.valves.close(4)
                    time.sleep(self.ptd_dict[self.pre_lim]['rhl'][1])
                    
                elif self.status == 4: # Right high small
                    self.timer.restart()
                    self.valves.open(3)
                    self.valves.close(3)
                    time.sleep(self.ptd_dict[self.pre_lim]['rhs'][0])
                    self.valves.open(4)
                    self.valves.close(4)
                    time.sleep(self.ptd_dict[self.pre_lim]['rhs'][1])

                elif (self.status == 5) or (self.status == 8): # temp: right too low / rnf
                    self.timer.restart()
                    self.valves.open(1)
                    time.sleep(self.ptd_dict[self.pre_lim]['rtl'][1])
                    self.valves.close(1)
                    self.valves.open(2)
                    time.sleep(self.ptd_dict[self.pre_lim]['rtl'][1])
                    self.valves.close(2)
                    time.sleep(self.ptd_dict[self.pre_lim]['rtl'][0])
                    
                
                elif (self.status == 6) or (self.status == 9): # temp: left too low / lnf
                    self.timer.restart()
                    self.valves.open(3)
                    time.sleep(self.ptd_dict[self.pre_lim]['ltl'][1])
                    self.valves.close(3)
                    self.valves.open(4)
                    time.sleep(self.ptd_dict[self.pre_lim]['ltl'][1])
                    self.valves.close(4)
                    time.sleep(self.ptd_dict[self.pre_lim]['ltl'][0])

                elif self.status == 7:
                    self.timer.start()
                    time.sleep(0.2)
                    if self.timer.get_elap_time() >= self.elapsed_time:
                        self.stop()
                        self.mp.alert('fe', f'{self.pressure:.2f}')

        except Exception as e:
            self.mp.send('de', f'{e}')

    def run(self) -> None:
        detect_thread = threading.Thread(target=self._run_detect)
        detect_thread.daemon = True
        detect_thread.start()

    def stop(self) -> None:
        try:
            self.mp.log('st')
            self.running = False
            self.valves.release()
            self.status = 0
            self.last_status = 10
            self.timer.stop()
        except Exception as e:
            self.mp.send('te', f'{e}')

    def stop_vacuum(self) -> None:
        try:
            self.vacuum_cache_running = False
            self.vacuum_reached_since = None
            self.mp.log('vt')
            self.valves.close(5)
        except Exception as e:
            self.mp.send('ae', f'{e}')

    def stop_vacuum_2(self) -> None:
        try:
            self.mp.log('vt')
            self.valves.close(2)
            self.valves.close(1)
        except Exception as e:
            self.mp.send('ae', f'{e}')

    def stop_airin(self) -> None:
        try:
            self.airin = False
            self.mp.log('vt')
            self.valves.close(4)
            self.valves.close(3)
        except Exception as e:
            self.mp.send('ae', f'{e}')

    def release(self) -> None:
        try:
            self.mp.log('rl')
            self.vacuum_cache_running = False
            self.vacuum_reached_since = None
            self.valves.release()
            self.valves.release_port()
            self.running = False
            self.status = 0
            self.last_status = 10
            self.pressure = 0
        except Exception as e:
            self.mp.send('te', f'{e}')
