import os
file_name = "CNN_BASE.py"

for cnt in range(1,6 + 1):
    with open(file_name, encoding="utf-8") as f:
        data_lines = f.read()

    if cnt != 1:
        # 文字列置換
        data_lines = data_lines.replace("dataset_count = " + str(cnt-1), "dataset_count = " + str(cnt))

    # 同じファイル名で保存
    with open(file_name, mode="w", encoding="utf-8") as f:
        f.write(data_lines)
    os.system('python CNN_BASE.py')