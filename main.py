# 氨基甲酸铵分解平衡常数测定控制程序的主程序
# 项目作者：李峙德，刘一弘
# 文件作者：李峙德
# 邮箱：contact@chemview.net
# 最后更新：2026-06-26
# Main program for the ammonium carbamate decomposition equilibrium constant experiment
# Project authors Li Zhide and Liu Yihong
# File author Li Zhide
# Email contact@chemview.net
# Last updated 2026-06-26
import cv2
import time
import threading
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import webview
import numpy as np
import os
import json
import logging
from flask_cors import CORS
import level_detect
import message_process

class Webview:
    def __init__(self, dec_instance):
        self.d=dec_instance
        self.app = Flask(__name__, template_folder='web', static_folder='web')
        CORS(self.app)
        log = logging.getLogger('werkzeug')
        log.disabled = True
        self.setup_routes()
        self.config_file = 'config.json'
        self.load_config()
        self.main_window = None
        self.frame_rate = 25.0
        self.frame_interval = 1.0 / self.frame_rate
        self.last_time = time.time()
        
    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    if hasattr(self.d, key):
                        setattr(self.d, key, value)
    
    def save_config(self, config_data):
        # Read the saved config first
        existing_config = {}
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                existing_config = json.load(f)
        
        # Merge the incoming values
        merged_config = {**existing_config,** config_data}
        
        # Save the merged config
        with open(self.config_file, 'w') as f:
            json.dump(merged_config, f, indent=4)
        
        # Push the changed values into the detector
        for key, value in config_data.items():
            if hasattr(self.d, key):
                setattr(self.d, key, value)

    def get_merged_config(self):
        defaults = {
            'cap_num': getattr(self.d, 'cap_num', 0),
            'port': getattr(self.d, 'port', None),
            'gaup': getattr(self.d, 'gaup', None),
            'delay_large': getattr(self.d, 'delay_large', 0.5),
            'hsv_lower1': getattr(self.d, 'hsv_lower1', '[0, 100, 50]'),
            'hsv_upper1': getattr(self.d, 'hsv_upper1', '[30, 255, 255]'),
            'hsv_lower2': getattr(self.d, 'hsv_lower2', '[120, 100, 50]'),
            'hsv_upper2': getattr(self.d, 'hsv_upper2', '[200, 255, 255]'),
            'large_diff_threshold': getattr(self.d, 'large_diff_threshold', 50),
            'small_diff_threshold': getattr(self.d, 'small_diff_threshold', 5),
            'bottom_threshold': getattr(self.d, 'bottom_threshold', 0.85),
            'usemask': getattr(self.d, 'usemask', False),
            'elapsed_time': getattr(self.d, 'elapsed_time', 30),
            'vacuum_threshold': getattr(self.d, 'vacuum_threshold', -90),
        }
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                defaults.update(json.load(f))
        return defaults

    def get_current_frame(self):
        if hasattr(self.d, 'utubedet') and hasattr(self.d.utubedet, 'display_frame'):
            frame = self.d.utubedet.display_frame
            if frame is not None and frame.size > 0:
                # Keep the frame rate steady
                current_time = time.time()
                sleep_time = self.frame_interval - (current_time - self.last_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.last_time = time.time()
                return frame
        return None # Return None when no frame is ready

    def generate_frames(self):
        while True:
            try:
                # Try to fetch the latest detector frame
                frame = self.get_current_frame()
                
                if frame is not None:
                    # Keep the stream light enough for the UI
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Encode the frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 50])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        try: 
                            yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        except Exception as e:
                            pass
                else:
                    # Use a black frame while the camera is unavailable
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    ret, buffer = cv2.imencode('.jpg', blank_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.04)
                
            except Exception as e:
                self.d.mp.log('ve',f'{e}')
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, 'Camera Error', (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
    
    def setup_routes(self):
        
        @self.app.route('/')
        def index():
            return render_template('index_en.html')
        
        @self.app.route('/config')
        def config():
            return render_template('config_en.html')
        
        @self.app.route('/debug')
        def debug():
            return render_template('debug_en.html')
        
        @self.app.route('/about')
        def about():
            return render_template('about_en.html')
        
        @self.app.route('/api/status')
        def get_status():
            elapsed_time = 0
            if getattr(self.d.timer, 'started', False):
                elapsed_time = self.d.timer.get_elap_time()
            return jsonify({
                'message': getattr(self.d.mp, 'message', ''),
                'pressure': f"{getattr(self.d, 'pressure', 0):.2f} kPa",
                'vacuum_threshold': getattr(self.d, 'vacuum_threshold', -90),
                'elapsed_time': getattr(self.d, 'elapsed_time', 30),
                'current_elapsed_time': elapsed_time,
                'running': getattr(self.d, 'running', False)
            })

        @self.app.route('/api/set_pressure', methods=['POST'])
        def set_pressure():
            try:
                pressure = float(request.json.get('pressure'))
                self.d.vacuum_threshold = pressure
                self.save_config({'vacuum_threshold': pressure})
                return jsonify({'success': True, 'vacuum_threshold': pressure})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/set_elapsed_time', methods=['POST'])
        def set_elapsed_time():
            try:
                elapsed_time = float(request.json.get('elapsed_time'))
                if elapsed_time <= 0:
                    raise ValueError('elapsed_time must be greater than 0')
                self.d.elapsed_time = elapsed_time
                self.save_config({'elapsed_time': elapsed_time})
                return jsonify({'success': True, 'elapsed_time': elapsed_time})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/start', methods=['POST'])
        def start_eq():
            self.d.run()
            return jsonify({'success': True})
        
        @self.app.route('/api/stop', methods=['POST'])
        def stop_eq():
            self.d.stop()
            return jsonify({'success': True})
        
        @self.app.route('/api/save_config', methods=['POST'])
        def save_config():
            try:
                config_data = request.json
                self.save_config(config_data)
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/open_help')
        def open_help():
            try:
                help_file = 'help.pdf'
                if os.path.exists(help_file):
                    os.startfile(help_file)
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/open_log')
        def open_log():
            try:
                console_file = 'dec.log'
                if os.path.exists(console_file):
                    os.startfile(console_file)
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
            
        @self.app.route('/api/reload')
        def reload():
            self.d.release()
            self.d.con()
            return jsonify({'success': True})
        
        @self.app.route('/api/go_vacuum')
        def go_vacuum():
            self.d.go_vacuum()
            return jsonify({'success': True})
        
        @self.app.route('/api/go_vacuum_2')
        def go_vacuum_2():
            self.d.go_vacuum_2()
            return jsonify({'success': True})

        @self.app.route('/api/go_vacuum_3')
        def go_vacuum_3():
            self.d.go_vacuum_3()
            return jsonify({'success': True})
        
        @self.app.route('/api/airin')
        def go_airin():
            self.d.go_airin()
            return jsonify({'success': True})
        
        @self.app.route('/api/stop_vacuum')
        def stop_vacuum():
            self.d.stop_vacuum()
            return jsonify({'success': True})
        
        @self.app.route('/api/stop_vacuum_2')
        def stop_vacuum_2():
            self.d.stop_vacuum_2()
            return jsonify({'success': True})
        
        @self.app.route('/api/stop_airin')
        def stop_airin():
            self.d.stop_airin()
            return jsonify({'success': True})
            
        @self.app.route('/api/get_config')
        def get_config():
            try:
                return jsonify(self.get_merged_config())
            except Exception as e:
                return jsonify({'error': str(e)})
            
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
        
    def run_flask(self):
        self.app.run(host='127.0.0.1', port=18917, debug=False, use_reloader=False)

    def run(self):
        flask_thread = threading.Thread(target=self.run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        
        time.sleep(0.1)
        
        self.d.con()

        self.main_window = webview.create_window(
            'Determination of Equilibrium Constant',
            'http://127.0.0.1:18917/',
            width=1000,
            height=700,
            resizable=True,
            maximized=True
        )
        
        # Start webview
        webview.start()

if __name__=='__main__':
    try:
        d=level_detect.Detector()
        d.mp=message_process.MessageProcessor()
        webview_app=Webview(d)
        webview_app.run()
    except Exception as e:
        print(f'Error：{e}')
