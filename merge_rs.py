from utils import *
import numpy as np
np.set_printoptions(linewidth=400, precision=2)

h = load_from_pickle("reward_shaping_large.p")
r = load_from_pickle("reward_shaping_row.p")

x = h + r
print(x.T)
save_to_pickle(x, "rs_large2.p")