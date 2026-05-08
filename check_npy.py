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
            if obj.ndim == 2 and obj.shape[0] > 0 and np.issubdtype(obj.dtype, np.number):
                arr = obj.astype(np.float64)
                print(f"{prefix}   ├─ min/max: {np.nanmin(arr):.6g} / {np.nanmax(arr):.6g}")
                print(f"{prefix}   ├─ first  : {np.array2string(arr[0], precision=4, suppress_small=True)}")
                if arr.shape[1] == 16:
                    left_quat_norm = np.linalg.norm(arr[:, 3:7], axis=1)
                    right_quat_norm = np.linalg.norm(arr[:, 11:15], axis=1)
                    left_qerr = np.nanmedian(np.abs(left_quat_norm - 1.0))
                    right_qerr = np.nanmedian(np.abs(right_quat_norm - 1.0))
                    left_gripper_ok = np.nanmean((arr[:, 7] >= -0.05) & (arr[:, 7] <= 1.05))
                    right_gripper_ok = np.nanmean((arr[:, 15] >= -0.05) & (arr[:, 15] <= 1.05))
                    looks_endpose = (
                        left_qerr < 0.08
                        and right_qerr < 0.08
                        and left_gripper_ok > 0.9
                        and right_gripper_ok > 0.9
                    )
                    guess = "endpose(末端位姿)" if looks_endpose else "qpos/joint(关节) 或其他"
                    print(f"{prefix}   ├─ 16D猜测: {guess}")
                    print(
                        f"{prefix}   ├─ quat误差: left={left_qerr:.6g}, right={right_qerr:.6g}; "
                        f"gripper[0,1]比例: left={left_gripper_ok:.2%}, right={right_gripper_ok:.2%}"
                    )
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
