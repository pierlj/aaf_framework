import logging
import torch
import numpy as np

def apply_tensor_list(tensor_list, method_name, *args, **kwargs):
    # Helper function to apply tensor methods and slicing on List[Tensor]
    if method_name[0] == '[':
        snippet = 'temp = t{}'.format(method_name)
    else:
        if len(args) > 0 and len(kwargs) > 0:
            args_str = join_list(args) + ',' + join_dict(kwargs, ',')
        elif len(args) > 0:
            args_str = join_list(args)
        elif len(kwargs) > 0:
            args_str = join_dict(kwargs, ',')
        else:
            args_str = ''
        snippet = 'temp = t.{}({})'.format(method_name, args_str)

    res_list = []
    temp = None
    for t in tensor_list:
        loc = locals()
        exec(snippet, globals(), loc)
        res_list.append(loc['temp'])
    return res_list

# def apply_tensor_list(tensor_list, method_name, *args, **kwargs):
#     def apply_tensor(tensor, method_name, *args, **kwargs):
#         fn = getattr(tensor, method_name)
#         return fn(*args, **kwargs)

#     return [apply_tensor(t, method_name, *args, **kwargs) for t in tensor_list]


def join_dict(d, sep=','):
    dict_list = ['{}={}'.format(k, v) for k, v in d.items()]
    return sep.join(dict_list)


def join_list(l, sep=','):
    dict_list = ['{}'.format(v) for v in l]
    return sep.join(dict_list)


class DisableLogger():
    """
    Context manager to temporarily disable info chennel in logger
    """
    def __enter__(self):
        logging.disable(logging.INFO)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def find_id_in_coco_img(coco, attr, value):
    for img in coco.imgs.values():
        if img[attr] == value:
            return img['id']

    return -1


def random_choice(len_tensor, n, generator=None):
    keep = torch.randperm(len_tensor, generator=generator)[:n]
    if len_tensor < n:
        pad = n - len_tensor
        keep_pad = torch.from_numpy(
            np.random.choice(np.arange(len_tensor), pad))
        keep = torch.cat([keep, keep_pad])
    return keep