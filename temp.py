#!/usr/bin/env python3
"""
CLI 工具：将文件中所有形如 xxx0xx 的6位数字键替换为 xxx1xx
即将第4位数字从 0 改为 1。

用法:
    python replace_key_digit.py <input_file> [output_file]

参数:
    input_file   输入文件路径
    output_file  输出文件路径（可选，默认覆盖原文件）

示例:
    python replace_key_digit.py data.json
    python replace_key_digit.py data.json data_new.json
"""

import re
import sys
import shutil
from pathlib import Path


PATTERN = re.compile(r'(\d{3})0(\d{2})')
REPLACEMENT = r'\g<1>1\2'


def replace_in_text(text: str) -> tuple[str, int]:
    """替换文本中所有匹配项，返回 (新文本, 替换次数)"""
    count = 0

    def replacer(m):
        nonlocal count
        count += 1
        return m.group(1) + '1' + m.group(2)

    new_text = PATTERN.sub(replacer, text)
    return new_text, count


def main():
    args = sys.argv[1:]

    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    input_path = Path(args[0])
    if not input_path.exists():
        print(f"[ERROR] 文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args[1]) if len(args) >= 2 else None

    # 读取文件
    try:
        text = input_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 执行替换
    new_text, count = replace_in_text(text)

    if count == 0:
        print("[INFO] 未找到任何匹配项（xxx0xx），文件未修改。")
        sys.exit(0)

    # 写出文件
    if output_path is None:
        # 覆盖原文件前先备份
        backup_path = input_path.with_suffix(input_path.suffix + '.bak')
        shutil.copy2(input_path, backup_path)
        print(f"[INFO] 已备份原文件至: {backup_path}")
        output_path = input_path

    try:
        output_path.write_text(new_text, encoding='utf-8')
    except Exception as e:
        print(f"[ERROR] 写入文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] 共替换 {count} 处，结果已写入: {output_path}")


if __name__ == '__main__':
    main()