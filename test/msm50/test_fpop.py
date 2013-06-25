from fpop import FPOPExchanger
import numpy as np
from scipy.io import mmread
from msmbuilder import Trajectory
import IPython

t = mmread('tProb.mtx')

pdb = Trajectory.load_from_pdb('proteinG_wt.rename.pdb')

avg_sasas = np.loadtxt('avg_sasas.dat')

exchanger = FPOPExchanger(t, avg_sasas, 100E-6, 1E-3, 20E-3, pdb, lagtime=50E-9)

for i in range(100):

    print i
    exchanger.next_step()

IPython.embed()
