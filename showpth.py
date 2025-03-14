import torch
from pprint import pprint


def show_pth_keys(file_path):
    # 加载pth文件（自动检测设备）
    data = torch.load(file_path, map_location='cpu')

    print(f"\n[File Structure] Total keys: {len(data)}")
    pprint(list(data.keys()))

    print("\n[Detailed Contents]")
    for i, (key, value) in enumerate(data.items(), 1):
        print(f"{i}. Key: {key}")
        print(f"   Type: {type(value)}")

        # 数据类型分析
        if isinstance(value, torch.Tensor):
            print(f"   Shape: {value.shape}")
            print(f"   Dtype: {value.dtype}")
            print(f"   Values (sample):\n{value[:2] if len(value) > 2 else value}\n")
        elif isinstance(value, dict):
            print(f"   Sub-keys: {list(value.keys())[:5]}... (Total: {len(value)})")
        elif isinstance(value, (list, tuple)):
            print(f"   Length: {len(value)}")
            print(f"   First element type: {type(value[0]) if len(value) > 0 else None}")
        else:
            print(f"   Content: {str(value)[:200]}...\n")

# show_pth_keys('semantic/imagenet_semantic_clip_gpt.pth')
show_pth_keys('center_MiniImageNet_resnet.pth')