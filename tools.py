import math
import datetime
import json
import os
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS # <--- 导入 FAISS
from langchain.agents import Tool
# 注：直接使用 @tool 装饰器，它会返回一个 LangChain Tool 对象

CALENDAR_FILE = "calendar.json" # 定义日历文件的名称
ZHIPUAI_API_KEY = "2c2a258f6daf4ab1a7eaa1f1298f5a0d.Lajx17gMNJr4bk9F"
VECTOR_DB_DIR = "faiss_index_db" # <--- 确保目录名匹配

# --- RAG 工具 ---
try:
    print("正在加载本地 FAISS 索引...")
    embeddings = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY)
    vector_store = FAISS.load_local(
        VECTOR_DB_DIR, 
        embeddings,
        allow_dangerous_deserialization=True # <--- 注意：FAISS 加载需要这个参数
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    print("FAISS 索引加载成功！")
    RAG_ENABLED = True
except Exception as e:
    print(f"警告：加载 FAISS 索引失败: {e}。RAG 工具将不可用。")
    RAG_ENABLED = False

@tool
def calculator(input_str: str) -> str:
    """
    执行基本的数学运算。
    输入格式为 '数字1,数字2,运算符'。
    支持的运算符包括：+ (加), - (减), * (乘), / (除), ^ (乘方)。
    例如: '3,4,+' 返回 '7.0'; '10,2,/' 返回 '5.0'; '2,3,^' 返回 '8.0'。
    """
    try:
        # 清洗输入，移除常见干扰字符
        cleaned_input = input_str.strip().replace("'", "").replace("\"", "").replace("\n", "").replace("Observation:", "")
        
        parts = [x.strip() for x in cleaned_input.split(',') if x.strip()]
        
        if len(parts) != 3:
            return "输入格式错误：请提供三个用逗号分隔的部分：'数字1,数字2,运算符'。"

        num1_str, num2_str, operator = parts

        # 尝试将数字部分转换为浮点数，以支持小数运算
        num1 = float(num1_str)
        num2 = float(num2_str)

        # 根据运算符执行计算
        if operator == '+':
            result = num1 + num2
        elif operator == '-':
            result = num1 - num2
        elif operator == '*':
            result = num1 * num2
        elif operator == '/':
            if num2 == 0:
                return "错误：除数不能为零。"
            result = num1 / num2
        elif operator == '^':
            result = math.pow(num1, num2)
        else:
            return f"错误：不支持的运算符 '{operator}'。请使用 +, -, *, /, ^ 中的一个。"

        # 返回字符串格式的结果
        return str(result)

    except ValueError:
        return "输入错误：无法将输入转换为数字。请确保前两个部分是有效的数字。"
    except Exception as e:
        return f"计算时发生未知错误：{str(e)}"
    
@tool
def write_to_file(input_str: str) -> str:
    """
    将内容写入到当前目录下的指定文件中。如果文件已存在，则覆盖。
    输入格式为 '文件名,要写入的内容'。
    例如: 'myfile.txt,这是要写入的内容。'
    """
    try:
        parts = input_str.split(',', 1) # 只按第一个逗号分割
        if len(parts) != 2:
            return "输入格式错误：请使用 '文件名,要写入的内容' 格式。"
        
        filename, content = parts[0].strip(), parts[1].strip()
        filename = filename.strip().replace("'", "").replace("\"", "").replace("\n", "").replace("Observation:", "")
        content = content.strip().replace("Observation:", "")

        # 安全检查：确保文件名不包含路径信息，防止写入到意外位置
        if '/' in filename or '\\' in filename or '..' in filename:
            return "错误：文件名不能包含路径信息。"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件 '{filename}' 已成功写入。"
    except Exception as e:
        return f"写入文件时发生错误: {str(e)}"

@tool
def read_from_file(filename: str) -> str:
    """
    读取当前目录下指定文件的内容。
    输入是要读取的文件名，例如 'myfile.txt'。
    """
    filename = filename.strip().replace("'", "").replace("\"", "").replace("\n", "").replace("Observation:", "")
    try:
        # 安全检查
        if '/' in filename or '\\' in filename or '..' in filename:
            return "错误：文件名不能包含路径信息。"
            
        if not os.path.exists(filename):
            return f"错误：文件 '{filename}' 不存在。"

        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return content if content else "文件为空。"
    except Exception as e:
        return f"读取文件时发生错误: {str(e)}"

@tool
def get_current_datetime(input_str: str) -> str:
    """
    获取当前的日期、时间和星期几。
    此工具 **逻辑上不需要输入**，但为了兼容性，调用时 **必须** 提供一个任意的非空字符串作为占位符，例如输入 '1'。
    返回格式如 '2025-05-26 11:52:47, Monday'。
    """
    now = datetime.datetime.now()
    # %A 会根据系统的 locale 返回星期几的完整名称 (可能是英文或中文)
    return now.strftime("%Y-%m-%d %H:%M:%S, %A")

def _load_calendar() -> dict:
    """加载日历文件，如果不存在则返回空字典。"""
    if not os.path.exists(CALENDAR_FILE):
        return {}
    try:
        with open(CALENDAR_FILE, 'r', encoding='utf-8') as f:
            # 如果文件为空，也返回空字典
            content = f.read()
            return json.loads(content) if content else {}
    except (json.JSONDecodeError, IOError):
        # 如果文件损坏或读取错误，返回空字典（或进行错误处理）
        return {}

def _save_calendar(data: dict):
    """将日历数据保存到文件。"""
    with open(CALENDAR_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

@tool
def add_calendar_event(input_str: str) -> str:
    """
    向日历中添加一个待办事项。
    输入格式为 'YYYY-MM-DD,待办事项描述'。
    例如: '2025-12-25,准备圣诞礼物'。
    """
    try:
        cleaned_input = input_str.strip().replace("'", "").replace("\"", "").replace("\n", "").replace("Observation:", "")
        parts = cleaned_input.split(',', 1)
        if len(parts) != 2:
            return "输入格式错误：请使用 'YYYY-MM-DD,待办事项描述' 格式。"
        
        date_str, event = parts[0].strip(), parts[1].strip()
        
        # 简单的日期格式验证
        datetime.datetime.strptime(date_str, '%Y-%m-%d')

        calendar_data = _load_calendar()
        if date_str not in calendar_data:
            calendar_data[date_str] = []
        
        calendar_data[date_str].append(event)
        _save_calendar(calendar_data)
        
        return f"已将 '{event}' 添加到 {date_str} 的日程中。"
    except ValueError:
        return "日期格式错误：请使用 YYYY-MM-DD 格式。"
    except Exception as e:
        return f"添加日历事件时发生错误: {str(e)}"

@tool
def get_calendar_events(date_str: str) -> str:
    """
    查询指定日期的待办事项。
    输入格式为 'YYYY-MM-DD,read'。例如: '2025-12-25,read'。
    """
    try:
        date_str = date_str.strip().replace("'", "").replace("\"", "").replace(",read", "").replace("\n", "").replace("Observation:", "")
        datetime.datetime.strptime(date_str, '%Y-%m-%d') # 验证格式

        calendar_data = _load_calendar()
        events = calendar_data.get(date_str, [])
        
        if not events:
            return f"{date_str} 没有安排任何事项。"
        else:
            event_list = "\n".join([f"- {e}" for e in events])
            return f"{date_str} 的日程安排如下:\n{event_list}"
            
    except ValueError:
        return "日期格式错误：请使用 YYYY-MM-DD 格式。"
    except Exception as e:
        return f"查询日历事件时发生错误: {str(e)}"

@tool
def knowledge_base_search(query: str) -> str:
    """
    当需要查询关于 [您的知识库主题，例如 '项目内部资料', '我的学习笔记', '文献库'] 的信息时使用此工具。
    输入用户的问题或查询关键词。
    返回从本地知识库中检索到的最相关信息。
    """
    if not RAG_ENABLED:
        return "错误：本地知识库工具当前不可用。"
    try:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "在本地知识库中没有找到相关信息。"
        
        context = "\n---\n".join([f"来源: {doc.metadata.get('source', 'N/A')}\n内容: {doc.page_content}" for doc in docs])
        return f"从知识库中找到以下相关信息：\n{context}"
    except Exception as e:
        return f"检索知识库时发生错误: {str(e)}"

# 初始化 DuckDuckGo 搜索工具
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="当需要回答关于当前事件、世界知识或不确定的问题时，执行网页搜索获取最新信息。"
)