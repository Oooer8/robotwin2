import numpy as np
import sys

def inspect_npy(filepath):
    print(f"\n{'='*50}")
    print(f"📂 文件路径: {filepath}")
    print(f"{'='*50}")

    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    def describe(obj, name="root", indent=0):
        prefix = "  " * indent
        if isinstance(obj, np.ndarray):
            print(f"{prefix}📦 [{name}]")
            print(f"{prefix}   ├─ type   : numpy.ndarray")
            print(f"{prefix}   ├─ dtype  : {obj.dtype}")
            print(f"{prefix}   ├─ shape  : {obj.shape}")
            print(f"{prefix}   └─ ndim   : {obj.ndim}")
            # 如果是 object 数组，递归展开
            if obj.dtype == object and obj.ndim > 0:
                print(f"{prefix}   └─ (object array, 展开第一个元素...)")
                try:
                    describe(obj.flat[0], name=f"{name}[0]", indent=indent+2)
                except Exception:
                    pass
        elif isinstance(obj, dict):
            print(f"{prefix}📋 [{name}] dict, keys({len(obj)}):")
            for k, v in obj.items():
                describe(v, name=str(k), indent=indent+1)
        elif isinstance(obj, (list, tuple)):
            type_name = type(obj).__name__
            print(f"{prefix}📝 [{name}] {type_name}, len={len(obj)}")
            if len(obj) > 0:
                describe(obj[0], name=f"{name}[0]", indent=indent+1)
        else:
            print(f"{prefix}🔹 [{name}] type={type(obj).__name__}, value={repr(obj)[:80]}")

    describe(data)
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 交互式输入
        path = input("请输入 .npy 文件路径: ").strip()
        inspect_npy(path)
    else:
        # 支持批量传入多个路径
        for path in sys.argv[1:]:
            inspect_npy(path)