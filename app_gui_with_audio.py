import tkinter as tk
from tkinter import scrolledtext, messagebox, Label
import threading
import queue
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
# 导入 tools 模块
try:
    from tools import calculator, write_to_file, read_from_file, get_current_datetime, add_calendar_event, get_calendar_events, knowledge_base_search, search_tool
except ImportError as e:
    messagebox.showerror("错误", e)
    exit()
# 导入 ASR 控制器
from audio_handler import XunfeiASRController

# --- 讯飞语音识别apikey ---
XUNFEI_APPID = "e7b66bd6"
XUNFEI_APIKEY = "01f9f321b05d375044f9c24a75205acc"
XUNFEI_APISECRET = "MDI3MWZkNGMzOWQyOGFkZGRiZDFlNGRh"

# --- Zhipu GLM apikey ---
OPENAI_API_KEY = "2c2a258f6daf4ab1a7eaa1f1298f5a0d.Lajx17gMNJr4bk9F"
OPENAI_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"
OPENAI_MODEL = "glm-4-air"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LangChain智能助手(支持语音输入)")
        self.root.geometry("1000x800")

        self.llm = None
        self.agent_with_memory = None
        self.message_queue = queue.Queue()
        self.asr_queue = queue.Queue() # <--- ASR 结果队列

        # --- ASR 控制器 ---
        self.asr_controller = XunfeiASRController(XUNFEI_APPID, XUNFEI_APIKEY, XUNFEI_APISECRET, self.asr_queue)
        
        self.setup_langchain()
        self.setup_gui()

        # 启动队列检查器
        self.root.after(100, self.check_queues) # 改为检查多个队列

    def setup_langchain(self):
        """初始化 LangChain 模型和 Agent"""
        try:
            self.llm = ChatOpenAI(
                model_name=OPENAI_MODEL, temperature=0.7,
                openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE
            )

            agent_tools = [
                calculator, 
                write_to_file,
                read_from_file, 
                get_current_datetime, 
                add_calendar_event, 
                get_calendar_events,
                knowledge_base_search,
                search_tool
            ]

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            self.agent_with_memory = initialize_agent(
                tools=agent_tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True, # 在控制台打印详细过程，方便调试
                handle_parsing_errors=True # 增强鲁棒性
            )
            print("LangChain Agent 初始化成功！")
        except Exception as e:
            messagebox.showerror("LangChain 初始化错误", f"无法初始化 Agent: {e}")
            import traceback; traceback.print_exc(); self.root.quit()

    def setup_gui(self):
        """设置 Tkinter GUI 界面"""
        # --- 对话历史框 ---
        history_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        history_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.history_text = scrolledtext.ScrolledText(history_frame, wrap=tk.WORD, state='disabled', font=("Arial", 10))
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # --- ASR 状态标签 ---
        self.asr_status_label = Label(self.root, text="语音识别状态: 空闲", fg="gray")
        self.asr_status_label.pack(pady=2)

        # --- 输入框和按钮 ---
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        # --- ASR 按钮 ---
        self.start_rec_button = tk.Button(input_frame, text="🎤 开始录音", command=self.start_recording_clicked, bg="#4CAF50", fg="white")
        self.start_rec_button.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_rec_button = tk.Button(input_frame, text="⏹️ 停止录音", command=self.stop_recording_clicked, bg="#f44336", fg="white", state='disabled')
        self.stop_rec_button.pack(side=tk.LEFT, padx=(0, 10))
        
        if not self.asr_controller:
            self.start_rec_button.config(state='disabled')
            self.asr_status_label.config(text="语音识别状态: 未配置凭证")


        self.input_entry = tk.Entry(input_frame, font=("Arial", 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.input_entry.bind("<Return>", self.send_message_event)

        self.send_button = tk.Button(input_frame, text="发送", command=self.send_message_event, font=("Arial", 10))
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

    def start_recording_clicked(self):
        if not self.asr_controller: return
        self.asr_controller.start_recording()
        self.start_rec_button.config(state='disabled')
        self.stop_rec_button.config(state='normal')
        self.asr_status_label.config(text="语音识别状态: 正在录音...", fg="red")
        self.input_entry.delete(0, tk.END) # 清空输入框

    def stop_recording_clicked(self):
        if not self.asr_controller: return
        self.asr_controller.stop_recording()
        self.start_rec_button.config(state='normal')
        self.stop_rec_button.config(state='disabled')
        self.asr_status_label.config(text="语音识别状态: 处理中...", fg="orange")

    def display_message(self, who: str, message: str, tag: str, extra_tags=None):
        """在历史框中显示消息"""
        self.history_text.config(state='normal')
        start_index = self.history_text.index(tk.END + "-1c")
        self.history_text.insert(tk.END, f"{who}: ", (tag, 'bold'))
        self.history_text.insert(tk.END, f"{message}\n\n")
        end_index = self.history_text.index(tk.END + "-1c")
        if extra_tags:
            if isinstance(extra_tags, str): extra_tags = [extra_tags]
            for extra_tag in extra_tags:
                self.history_text.tag_add(extra_tag, start_index, end_index)
        self.history_text.config(state='disabled')
        self.history_text.see(tk.END)

    def send_message_event(self, event=None):
        """处理发送按钮点击或回车事件"""
        user_input = self.input_entry.get().strip()
        if not user_input or self.agent_with_memory is None: return
        self.input_entry.delete(0, tk.END)
        self.display_message("您", user_input, "user")
        self.display_message("助手", "正在思考中...", "thinking", extra_tags="thinking_msg")
        self.send_button.config(state='disabled')
        self.input_entry.config(state='disabled')
        threading.Thread(target=self.run_agent_thread, args=(user_input,), daemon=True).start()

    def run_agent_thread(self, user_input):
        """在后台线程中运行 Agent 并将结果放入队列"""
        try:
            response = self.agent_with_memory.run(user_input)
            self.message_queue.put(response)
        except Exception as e:
            error_message = f"抱歉，处理您的请求时出现错误: {e}"; print(f"Agent Error: {e}")
            self.message_queue.put(error_message)

    def check_queues(self): # 修改为检查两个队列
        """检查 Agent 和 ASR 队列并更新 GUI"""
        # 检查 Agent 队列
        try:
            while True:
                message = self.message_queue.get_nowait() 
                self.history_text.config(state='normal')
                thinking_ranges = self.history_text.tag_ranges('thinking_msg')
                if thinking_ranges:
                    self.history_text.delete(thinking_ranges[0], thinking_ranges[1])
                    self.history_text.tag_remove('thinking_msg', '1.0', tk.END) 
                self.history_text.config(state='disabled')
                self.display_message("助手", message, "agent")
                self.send_button.config(state='normal')
                self.input_entry.config(state='normal')
                self.input_entry.focus_set()
        except queue.Empty:
            pass 

        # 检查 ASR 队列
        try:
            while True:
                asr_message = self.asr_queue.get_nowait()
                if asr_message.startswith("STATUS:"):
                    self.asr_status_label.config(text=f"语音识别状态: {asr_message[7:]}")
                    if "结束" in asr_message or "错误" in asr_message:
                         self.start_rec_button.config(state='normal')
                         self.stop_rec_button.config(state='disabled')
                         self.asr_status_label.config(fg="gray")

                elif asr_message.startswith("ERROR:"):
                     self.asr_status_label.config(text=f"语音识别状态: {asr_message}", fg="red")
                     messagebox.showerror("ASR 错误", asr_message)
                else: # 是识别结果
                    self.input_entry.delete(0, tk.END)
                    self.input_entry.insert(0, asr_message.replace("...", "")) # 插入时去掉省略号
                    if not asr_message.endswith("..."):
                        # 如果是最终结果，可以自动发送 (可选)
                        # self.send_message_event()
                        pass
        except queue.Empty:
            pass

        self.root.after(100, self.check_queues) # 安排下一次检查

    def run(self):
        """启动 Tkinter 事件循环"""
        self.history_text.tag_config('user', foreground='blue')
        self.history_text.tag_config('agent', foreground='green')
        self.history_text.tag_config('thinking', foreground='gray', font=("Arial", 10, "italic"))
        self.history_text.tag_config('bold', font=("Arial", 10, "bold"))
        self.history_text.tag_config('thinking_msg')
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    app.run()