
# I just wanted this to be a folder so that it would look more like the other modules

import os


def open_file(filepath, binary_mode=False):
    mode = 'w' if os.path.exists(filepath) else 'x'
    if binary_mode:
        mode += 'b'
    return open(filepath, mode)
