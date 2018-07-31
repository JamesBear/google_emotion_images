
# V0.1 Updated 20180720

SAVE_DIR = 'saves_and_logs'

import datetime
import os
import pickle
import numpy as np
def load_object(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def create_save_files():
    global logfile, pickle_path, output_file
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile_path = os.path.join(SAVE_DIR, time_str + 'log.txt')
    logfile = open(logfile_path, 'w')
    pickle_path = os.path.join(SAVE_DIR, time_str+'.pkl')
    output_file = 'prediction'+time_str+'.csv'

    #print('log file: ', logfile_path)

import sys
def lprint(*objects, sep=' ', end='\n', flush=False):
    """
    Print to sys.stdout and also logfile.
    """
    print(*objects, sep=sep, file=sys.stdout, end=end, flush=flush)
    print(*objects, sep=sep, file=logfile, end=end, flush=True)

def log_ready():
    return 'logfile' in globals()

import time
class StopWatch:
    def __init__(self, print_func = None):
        if print_func != None:
            self.print_func = print_func
        elif log_ready():
            self.print_func = lprint
        else:
            self.print_func = print

    def start(self, msg=None):
        self.start_time = time.time()
        if msg != None:
            self.print_func(msg)

    def stop(self, msg=None):
        current = time.time()
        elapsed = current - self.start_time
        self.print_func("{} took {:.3f} seconds".format(msg, elapsed))


def init_log_utils():
    global watch
    create_save_files()
    watch = StopWatch(lprint)
    np.random.seed(42)
