
import os, time

class Logger:
    def __init__(self, logdir):
        os.makedirs(logdir, exist_ok=True)
        self.file = open(os.path.join(logdir, "train.log"), "a")
    def log(self, msg):
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"[{t}] {msg}\n")
        self.file.flush()
