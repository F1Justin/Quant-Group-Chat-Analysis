import os
import pandas as pd
import psycopg2
from multiprocessing import Pool

# 读取群组列表
groups_list = []
with open('groups_list.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) > 1 and parts[1].isdigit():  # 确保有群号且为数字
            groups_list.append((parts[0], parts[1]))  # 保存标识和群号

def export_group_data(args):
    name, group = args

    # 建立数据库连接
    conn = psycopg2.connect(
        dbname="botmsg",
        user="justin",
        password="",
        host="localhost",
        port="5432"
    )

    try:
        query = """
        SELECT
            m.id, m.time, m.type, m.plain_text, m.message, s.id2 AS group_id, s.id1 AS user_id
        FROM
            nonebot_plugin_chatrecorder_messagerecord m
        JOIN
            nonebot_plugin_session_orm_sessionmodel s
        ON
            m.session_persist_id = s.id
        WHERE
            s.id2 = %s;
        """
        # 导出为 CSV
        df = pd.read_sql_query(query, conn, params=(group,))
        output_path = os.path.join('.', f'origin_{name}_{group}.csv')
        df.to_csv(output_path, index=False)
    finally:
        # 确保连接关闭
        conn.close()

# 使用多进程并行导出
if __name__ == "__main__":
    with Pool(8) as p:  # 使用 8 个进程并行导出
        p.map(export_group_data, groups_list)