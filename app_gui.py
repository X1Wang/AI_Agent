import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import queue
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

try:
    from tools import calculator, write_to_file, read_from_file, get_current_datetime, add_calendar_event, get_calendar_events, knowledge_base_search, search_tool
except ImportError:
    messagebox.showerror("错误", "无法找到 'tools.py' 文件或其中的工具。请确保文件存在且工具已定义。")
    exit()

OPENAI_API_KEY = "2c2a258f6daf4ab1a7eaa1f1298f5a0d.Lajx17gMNJr4bk9F"
OPENAI_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"
OPENAI_MODEL = "glm-4-air"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LangChain智能助手")
        self.root.geometry("1000x800")

        self.llm = None
        self.agent_with_memory = None
        self.message_queue = queue.Queue()

        self.setup_langchain()
        self.setup_gui()

        # 启动队列检查器
        self.root.after(100, self.check_queue)

    def setup_langchain(self):
        """初始化 LangChain 模型和 Agent"""
        try:
            self.llm = ChatOpenAI(
                model_name=OPENAI_MODEL,
                temperature=0.7,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE
            )

            tools = [
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
                tools=tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True, # 在控制台打印详细过程，方便调试
                handle_parsing_errors=True # 增强鲁棒性
            )
            print("LangChain Agent 初始化成功！")
        except Exception as e:
            messagebox.showerror("LangChain 初始化错误", f"无法初始化 Agent: {e}\n请检查您的 API Key 和网络连接。")
            self.root.quit()

    def setup_gui(self):
        """设置 Tkinter GUI 界面"""
        # --- 对话历史框 ---
        history_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        history_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.history_text = scrolledtext.ScrolledText(history_frame, wrap=tk.WORD, state='disabled', font=("Arial", 10))
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # --- 输入框和按钮 ---
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.input_entry = tk.Entry(input_frame, font=("Arial", 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.input_entry.bind("<Return>", self.send_message_event) # 绑定回车键

        self.send_button = tk.Button(input_frame, text="发送", command=self.send_message_event, font=("Arial", 10))
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

    def display_message(self, who: str, message: str, tag: str, extra_tags=None):
        """在历史框中显示消息，并可添加额外标签"""
        self.history_text.config(state='normal')
        
        # 记录插入前的索引
        start_index = self.history_text.index(tk.END + "-1c") # -1c 确保在末尾隐式换行符之前

        # 插入内容
        self.history_text.insert(tk.END, f"{who}: ", (tag, 'bold'))
        self.history_text.insert(tk.END, f"{message}\n\n")

        # 记录插入后的索引
        end_index = self.history_text.index(tk.END + "-1c")

        # 如果有额外标签，应用到刚刚插入的整个范围
        if extra_tags:
            if isinstance(extra_tags, str):
                extra_tags = [extra_tags]
            for extra_tag in extra_tags:
                self.history_text.tag_add(extra_tag, start_index, end_index)

        self.history_text.config(state='disabled')
        self.history_text.see(tk.END) # 自动滚动到底部

    def send_message_event(self, event=None):
        """处理发送按钮点击或回车事件"""
        user_input = self.input_entry.get().strip()
        if not user_input or self.agent_with_memory is None:
            return

        self.input_entry.delete(0, tk.END)
        self.display_message("您", user_input, "user")
        self.display_message("助手", "正在思考中...", "thinking", extra_tags="thinking_msg")

        # 禁用按钮和输入框，防止重复发送
        self.send_button.config(state='disabled')
        self.input_entry.config(state='disabled')

        # 在新线程中运行 Agent
        threading.Thread(target=self.run_agent_thread, args=(user_input,), daemon=True).start()

    def run_agent_thread(self, user_input):
        """在后台线程中运行 Agent 并将结果放入队列"""
        try:
            response = self.agent_with_memory.run(user_input)
            self.message_queue.put(response)
        except Exception as e:
            error_message = f"抱歉，处理您的请求时出现错误: {e}"
            print(f"Agent Error: {e}") # 在控制台打印详细错误
            self.message_queue.put(error_message)

    def check_queue(self):
        """检查队列中是否有来自 Agent 的新消息，并更新 GUI (使用标签删除)"""
        try:
            while True:
                # 从队列中获取消息，如果队列为空，会引发 queue.Empty 异常
                message = self.message_queue.get_nowait() 

                # 启用文本框编辑
                self.history_text.config(state='normal')

                # 获取 "thinking_msg" 标签的所有范围
                thinking_ranges = self.history_text.tag_ranges('thinking_msg')

                # 如果找到了标签范围 (它会返回一个元组，如 (start1, end1, start2, end2,...))
                if thinking_ranges:
                    # 我们只需要删除第一个（也是唯一一个）范围
                    start_index = thinking_ranges[0]
                    end_index = thinking_ranges[1]
                    # 精确删除带有标签的文本
                    self.history_text.delete(start_index, end_index)
                    # 虽然删除文本会带走标签，但保险起见可以手动移除所有标签
                    self.history_text.tag_remove('thinking_msg', '1.0', tk.END) 

                # 禁用文本框编辑（在显示新消息之前或之后都可以，但显示前禁用更安全）
                self.history_text.config(state='disabled')

                # 显示 Agent 的真实回复 (注意：这里不再传递 extra_tags)
                self.display_message("助手", message, "agent")

                # 恢复按钮和输入框
                self.send_button.config(state='normal')
                self.input_entry.config(state='normal')
                self.input_entry.focus_set() # 让输入框重新获得焦点

        except queue.Empty:
            pass # 队列为空，直接跳过

        # 每 100 毫秒检查一次队列
        self.root.after(100, self.check_queue)

    def run(self):
        """启动 Tkinter 事件循环"""
        # 设置标签样式
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