import socket
try:
    from absl import flags
except ImportError:
    flags = None

if flags is not None:
    FLAGS = flags.FLAGS
    FLAGS(["train_sc.py"])
