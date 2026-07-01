# 氨基甲酸铵分解平衡常数测定控制程序的一部分，用于处理消息及日志
# 项目作者：李峙德，刘一弘
# 文件作者：李峙德
# 邮箱：contact@chemview.net
# 最后更新：2026-06-26
# Message and log handling for the ammonium carbamate decomposition equilibrium constant experiment
# Project authors Li Zhide and Liu Yihong
# File author Li Zhide
# Email contact@chemview.net
# Last updated 2026-06-26

# send 
# alert 
# log 
# box 
import time
import pymsgbox
import plyer

class MessageProcessor:

    def __init__(self) -> None:
        self.message = None
        self.uimsg = {'wa':'Waiting', 're':'Ready', 'ce':'Hardware Connection Error', 'pe':'Preview Error', 'de':'Detection Error', 'ae':'Valve Error', 'te':'Stopping Error'}
        self.alertmsg = {'fe':'Equilibrium finished!!! Pressure'}
        self.boxmsg = {'se':'Starting equilibrium...'}
        self.logmsg = {'wa':'WAITING',
                     're':'READY',
                     'ce':'HWCONNECTIONERROR',
                     'pe':'PREVIEWERROR',
                     'de':'DETECTIONERROR',
                     'se':'STARTEQ',
                     'sc':'STATUSCHANGED',
                     'fe':'EQFINISHED',
                     've':'GENERATEVIDEOERROR',
                     'vs':'STARTVACUUM',
                     'ae':'VACUUMERROR',
                     'vt':'VACUUMSTOP',
                     'st':'STOPPED',
                     'rl':'RELEASE',
                     'te':'STOPERROR'
                     }
        self.title = 'Determination of Equilibrium Constant'

    def send(self, msg: str, d: str ='') -> None:
        try:
            if d:
                self.message=f'{self.uimsg[msg]}: {d}'
            else:
                self.message=self.uimsg[msg]
            self.log(msg, d)
        except Exception as e:
            print(e)

    def alert(self, msg: str, d: str ='') -> None:
        try:
            self.log(msg, d)
            if d:
                pymsgbox.alert(text=f'{self.alertmsg[msg]}: {d} kPa', title=self.title)
            else:
                pymsgbox.alert(text=self.alertmsg[msg], title=self.title)
        except Exception as e:
            print(e)

    def box(self, msg: str, d: str ='') -> None:
        try:
            self.log(msg, d)
            if d:
                plyer.notification.notify(
                    title=self.title,
                    message=f'{self.boxmsg[msg]}: {d}',
                    app_name=self.title,
                    timeout=3,
                )
            else:
                plyer.notification.notify(
                    title=self.title,
                    message=self.boxmsg[msg],
                    app_name=self.title,
                    timeout=3,
                )
        except Exception as e:
            print(e)

    def log(self, msg: str, d: str ='') -> None:
        try:
            t=time.strftime('%H:%M:%S', time.localtime())
            with open('dec.log', 'a') as file:
                file.write(f'\n{t} {self.logmsg[msg]} {d}')
        except Exception as e:
            print(e)

    def data(self, d: str ='') -> None:
        try:
            with open('data.csv', 'a') as file:
                file.write(f'\n{d}')
        except Exception as e:
            print(e)
