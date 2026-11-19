import sys
import os

SUBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CS3244-Fake-News-Detection-master")
if SUBDIR not in sys.path:
    sys.path.insert(0, SUBDIR)

from train_model import main

if __name__ == "__main__":
    main()

