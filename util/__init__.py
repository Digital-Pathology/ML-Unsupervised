
# I just wanted this to be a folder so that it would look more like the other modules

from itertools import repeat
import os


def open_file(filepath, mode='r', binary_mode=False):
    if mode in ['w', 'x']:
        mode = 'w' if os.path.exists(filepath) else 'x'
    if binary_mode:
        mode += 'b'
    return open(filepath, mode)


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    # https://stackoverflow.com/a/53173433/13747259
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    # https://stackoverflow.com/a/53173433/13747259
    return fn(*args, **kwargs)
