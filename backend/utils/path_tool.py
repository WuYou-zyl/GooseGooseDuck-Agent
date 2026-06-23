"""
为整个工程提供绝对路径
"""
import os

def get_root_path() -> str:
    """
    获取工程根目录路径
    """

    # backend/utils/path_tool.py -> parent of backend/ = repository root
    current_file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    return root_path

def get_abs_path(relative_path: str) -> str:
    """
    获取工程根目录下相对路径的绝对路径
    """
    root_path = get_root_path()
    abs_path = os.path.join(root_path, relative_path)
    return abs_path

if __name__ == '__main__':
    print(get_root_path())
    print(get_abs_path('data'))