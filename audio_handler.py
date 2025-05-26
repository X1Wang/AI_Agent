import websocket
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import threading # 使用 threading 模块
import pyaudio
import queue

STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2

class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn",
                             "accent": "mandarin", "vinfo": 1, "vad_eos": 2000} # vad_eos 调整长一点

    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'),
                                 signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        url = url + '?' + urlencode(v)
        return url

class XunfeiASRController:
    def __init__(self, app_id, api_key, api_secret, result_queue):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.result_queue = result_queue # 用于向 GUI 发送结果的队列
        self.ws_param = Ws_Param(self.app_id, self.api_key, self.api_secret)
        self.ws = None
        self.ws_thread = None
        self.audio_thread = None
        self.is_recording = threading.Event() # 使用 Event 来控制录音状态
        self.p_audio = None
        self.stream = None
        self.current_result = "" # 用于累积结果

    def _on_message(self, ws, message):
        try:
            message_data = json.loads(message)
            code = message_data["code"]
            sid = message_data["sid"]

            if code != 0:
                errMsg = message_data["message"]
                print(f"sid:{sid} call error:{errMsg} code is:{code}")
                self.result_queue.put(f"ERROR: {errMsg}")
            else:
                data = message_data["data"]["result"]["ws"]
                result = ""
                for i in data:
                    for w in i["cw"]:
                        result += w["w"]
                
                # 'pgs': 'apd' 表示追加，'rpl' 表示替换, 'rws' 表示结果
                pgs = message_data["data"]["result"].get("pgs")
                # 'ls': True 表示是最终结果
                ls = message_data["data"]["result"].get("ls", False)

                # 简单处理：如果是替换，就用新的；如果是追加，就加到后面
                # 讯飞的逻辑可能更复杂，这里做简化处理
                if pgs == 'rpl':
                     # 查找是否有历史句子（sn>1），替换最后一个
                    last_sentence = self.current_result.rfind("。")
                    if last_sentence != -1:
                        self.current_result = self.current_result[:last_sentence+1] + result
                    else:
                        self.current_result = result
                else: # 默认追加
                    self.current_result += result

                # 如果是最终结果，加上句号并清空以便下次累积
                if ls:
                    self.current_result += "。"
                    print(f"最终结果: {self.current_result}")
                    self.result_queue.put(self.current_result)
                    self.current_result = "" # 清空，准备下一句
                else:
                    # 发送部分结果
                    print(f"部分结果: {self.current_result}")
                    self.result_queue.put(self.current_result + "...") # 加省略号表示非最终

        except Exception as e:
            print(f"receive msg, but parse exception: {e}")
            self.result_queue.put(f"ERROR: 解析消息失败 - {e}")

    def _on_error(self, ws, error):
        print(f"### ASR error ### : {error}")
        self.result_queue.put(f"ERROR: ASR 连接错误 - {error}")
        self.stop_recording() # 尝试停止

    def _on_close(self, ws, close_status_code, close_msg):
        print("### ASR closed ###")
        # 确保 Pyaudio 停止
        if self.stream and self.stream.is_active():
             self.stream.stop_stream()
             self.stream.close()
        if self.p_audio:
            self.p_audio.terminate()
        self.p_audio = None
        self.stream = None

    def _on_open(self, ws):
        print("### ASR open ###")
        # Websocket 连接成功后，启动音频发送线程
        self.audio_thread = threading.Thread(target=self._send_audio_thread)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def _send_audio_thread(self):
        status = STATUS_FIRST_FRAME
        CHUNK = 520
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        try:
            self.p_audio = pyaudio.PyAudio()
            self.stream = self.p_audio.open(format=FORMAT, channels=CHANNELS,
                                             rate=RATE, input=True,
                                             frames_per_buffer=CHUNK)
            print("---------------开始录音-----------------")
            self.result_queue.put("STATUS: 开始录音...")

            while self.is_recording.is_set(): # 检查 Event 是否被设置
                buf = self.stream.read(CHUNK, exception_on_overflow = False) # 忽略溢出
                if not buf:
                    status = STATUS_LAST_FRAME
                
                if status == STATUS_FIRST_FRAME:
                    d = {"common": self.ws_param.CommonArgs,
                         "business": self.ws_param.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    status = STATUS_CONTINUE_FRAME
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                elif status == STATUS_LAST_FRAME:
                     d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                     break # 最后一帧后退出循环
                
                if self.ws and self.ws.sock and self.ws.sock.connected:
                    self.ws.send(json.dumps(d))
                else:
                    print("Websocket 已关闭，停止发送音频。")
                    break

            # 循环结束后（因为 is_recording 被清除），发送最后一帧信号
            if self.ws and self.ws.sock and self.ws.sock.connected:
                 last_frame_data = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                      "audio": "", "encoding": "raw"}}
                 self.ws.send(json.dumps(last_frame_data))
                 print("发送最后一帧信号...")
                 time.sleep(1) # 等待服务器处理

        except Exception as e:
            print(f"音频发送线程错误: {e}")
            self.result_queue.put(f"ERROR: 音频处理错误 - {e}")
        finally:
            print("---------------录音线程结束-----------------")
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
            if self.p_audio:
                self.p_audio.terminate()
            self.p_audio = None
            self.stream = None
            self.result_queue.put("STATUS: 录音结束。")
            # 在这里关闭 websocket 可能更好
            if self.ws:
                self.ws.close()


    def start_recording(self):
        if self.ws_thread and self.ws_thread.is_alive():
            print("录音已经在进行中。")
            return
            
        self.is_recording.set() # 设置 Event，表示可以录音
        self.current_result = "" # 清空历史结果
        wsUrl = self.ws_param.create_url()
        self.ws = websocket.WebSocketApp(wsUrl, on_message=self._on_message,
                                           on_error=self._on_error, on_close=self._on_close)
        self.ws.on_open = self._on_open
        
        # 在新线程中运行 websocket
        self.ws_thread = threading.Thread(target=lambda: self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_timeout=2))
        self.ws_thread.daemon = True
        self.ws_thread.start()
        print("ASR 控制器已启动。")

    def stop_recording(self):
        self.is_recording.clear() # 清除 Event，通知音频线程停止
        print("正在停止录音...")
        # 音频线程会在检测到 is_recording 为 False 后自行发送最后一帧并关闭 websocket
        # 我们不需要在这里显式关闭 websocket，让音频线程处理