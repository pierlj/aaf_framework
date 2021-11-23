from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


def is_notebook():
    """
    Find out if script is run in IPython or standard python interpreter
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

if is_notebook():
    mtqdm = tqdm_notebook
else:
    mtqdm = tqdm