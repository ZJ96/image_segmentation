import os

def write_log(log_file_name, log_messgae):
    if not os.path.exists("./log"):
        os.mkdir("./log")
    log_path_file_name = os.path.join("./log",log_file_name)
    with open(log_path_file_name, 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
        f.write(str(log_messgae))
        f.write("\n")
