import os
import shutil

def copy_matching_files(source_folder, compare_folder, target_folder):
    """
    根据比较文件夹中的文件名从源文件夹复制匹配文件到目标文件夹
    :param source_folder: 源文件所在文件夹 (J:\df\mutispectralunet\DATAnew\1\5)
    :param compare_folder: 比较文件夹 (J:\df\3\test)
    :param target_folder: 目标保存文件夹 (J:\df\5\test)
    """
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    
    # 获取比较文件夹中的文件名集合（不带扩展名）
    compare_files = set()
    for f in os.listdir(compare_folder):
        if os.path.isfile(os.path.join(compare_folder, f)):
            file_name = os.path.splitext(f)[0]  # 去除扩展名
            compare_files.add(file_name.lower())  # 不区分大小写
    
    # 遍历源文件夹
    copied_count = 0
    for src_file in os.listdir(source_folder):
        src_path = os.path.join(source_folder, src_file)
        
        # 跳过子文件夹
        if not os.path.isfile(src_path):
            continue
            
        # 获取文件名（不带扩展名）
        base_name = os.path.splitext(src_file)[0]
        
        # 检查是否存在于比较文件夹
        if base_name.lower() in compare_files:
            # 构建目标路径
            dst_path = os.path.join(target_folder, src_file)
            
            try:
                shutil.copy2(src_path, dst_path)
                print(f"已复制: {src_file}")
                copied_count += 1
            except Exception as e:
                print(f"复制失败 {src_file}: {str(e)}")
    
    print(f"\n操作完成！共找到 {copied_count} 个匹配文件")

# 使用示例
if __name__ == "__main__":
    # 设置路径（请根据实际情况修改）
    source_dir = r"J:\df\mutispectralunet\DATAnew\1\5"
    compare_dir = r"J:\df\3\val"
    target_dir = r"J:\df\5\val"
    
    # 执行复制操作
    copy_matching_files(source_dir, compare_dir, target_dir)