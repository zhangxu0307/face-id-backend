import os


def empty_dir(folder):
    """
    Empty a folder
    :param folder:
    :return:
    """
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def build_empty_dir(folder, root_dir=os.getcwd()):
    """
    Build (if required) and empty a directory
    :param folder:
    :param root_dir:
    :return:
    """
    os.makedirs(os.path.join(root_dir, folder), exist_ok=True)
    empty_dir(os.path.join(root_dir, folder))
