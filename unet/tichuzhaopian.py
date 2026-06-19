import os

def clean_mismatched_tifs(ref_folder, target_folder):
    """
    根据参考文件夹中的文件名清理目标文件夹
    :param ref_folder: 包含参考TIF文件的文件夹路径
    :param target_folder: 需要清理的目标文件夹路径
    """
    # 获取参考文件夹中的所有TIF文件名（不区分大小写）
    ref_files = set()
    for f in os.listdir(ref_folder):
        if f.lower().endswith(('.tif', '.tiff')):
            ref_files.add(f.lower())  # 统一转换为小写比较

    # 遍历目标文件夹
    deleted_count = 0
    kept_count = 0
    for filename in os.listdir(target_folder):
        file_path = os.path.join(target_folder, filename)
        
        # 仅处理TIF文件
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue
            
        # 检查文件名是否存在于参考集（不区分大小写）
        if filename.lower() not in ref_files:
            try:
                os.remove(file_path)
                print(f"🗑️ 已删除未匹配文件: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败 {filename}: {str(e)}")
        else:
            kept_count += 1

    # 输出统计结果
    print("\n清理完成:")
    print(f"✅ 保留文件: {kept_count} 个")
    print(f"🗑️ 删除文件: {deleted_count} 个")
    print(f"📂 目标文件夹剩余文件: {len(os.listdir(target_folder)) - deleted_count} 个")

# 使用示例
if __name__ == "__main__":
    # 设置路径（请根据实际情况修改）
    reference_folder = "G:/df/3/val"  # 参考文件夹
    target_folder = "G:/df/5/val"        # 需要清理的文件夹
    
    # 执行清理前确认
    confirm = input(f"即将清理 {target_folder}，确认继续？(y/n) ")
    if confirm.lower() == 'y':
        clean_mismatched_tifs(reference_folder, target_folder)
    else:
        print("操作已取消")