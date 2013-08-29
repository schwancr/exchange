from fpop import FPOPExchanger
import numpy as np
from scipy.io import mmread
from msmbuilder import Trajectory
from pylab import *

t = mmread('tProb.mtx')

pdb = Trajectory.load_from_pdb('proteinG_wt.rename.pdb')

avg_sasas = np.loadtxt('avg_sasas_noise5.dat')

timepoints = np.arange(0, 110, 10)[1:]

print timepoints.shape

exchanger = FPOPExchanger(t, avg_sasas, 100E-6, 1E-3, 20E-3, pdb, timepoints, lagtime=50E-9, init_pops=np.loadtxt('init48.dat'), force_dense=False)

total_products, total_res_pops = exchanger.run(return_res_pops=True, num_labels=10)

for i in range(4):
    plot(timepoints * 50, total_products[:, i])

xlabel('time (ns)')
show()
