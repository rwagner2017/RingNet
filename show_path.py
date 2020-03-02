import os
import sys

if __name__ == '__main__':
    print('PYTHONPATH')
    for p in sys.path:
        print(p)

    print()
    print('PATH')
    print(os.environ['PATH'])

    os.environ['PATH'] = '/venv/bin:/usr/lib/x86_64-linux-gnu:' + os.environ['PATH']

    print()
    print('new PATH')
    print(os.environ['PATH'])

    # os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu'
    print('LD_LIBRARY_PATH')
    print(os.environ['LD_LIBRARY_PATH'])

    # import tensorflow as tf