# utils/path_utils.py
import os
import sys


def get_project_root():
    """Get absolute path to project root directory"""
    # Find project root through current file path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)  # Parent of utils is project root
    return project_root


def ensure_absolute_path(path, relative_to_project=True):
    """Ensure path is absolute"""
    if os.path.isabs(path):
        return os.path.normpath(path)

    if relative_to_project:
        # Relative to project root
        project_root = get_project_root()
        return os.path.normpath(os.path.join(project_root, path))
    else:
        # Relative to current working directory
        return os.path.normpath(os.path.abspath(path))


def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    dir_path = ensure_absolute_path(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def safe_join(base_path, *paths):
    """Safe path joining with correct separators"""
    base_path = ensure_absolute_path(base_path)
    full_path = os.path.join(base_path, *paths)
    return os.path.normpath(full_path)


def find_file_in_project(filename):
    """Find file in project"""
    project_root = get_project_root()

    # First look in project root
    possible_paths = [
        os.path.join(project_root, filename),
        os.path.join(project_root, "config", filename),
        os.path.join(project_root, "data", filename),
        os.path.join(project_root, "models", filename),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # If not found, search entire project
    for root, dirs, files in os.walk(project_root):
        if filename in files:
            return os.path.join(root, filename)

    return None