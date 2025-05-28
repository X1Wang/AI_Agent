# 1.运行语音助手程序

终端目录切换到代码所在目录

安装python环境和必要的依赖库:

conda create --name agent python=3.10

pip install -r requirements.txt

然后执行：

python app_gui_with_audio.py

# 2.构建FAISS索引

先在knowledge_base文件夹内存储个人文档

然后运行python build_index.py