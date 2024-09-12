import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import time, timeit

import procedure_lib as proclib

ideal = proclib.Procedure(L=3)

data = ideal.run_with_mesolve()

