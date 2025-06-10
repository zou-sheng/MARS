with open('./layers.yaml', 'w') as f:
    # 生成并写入四种类型的文件名
    for param in ['C', 'K', 'P', 'R']:
        for i in range(1, 129):
            f.write(f"- problem_{param}{i}\n")
