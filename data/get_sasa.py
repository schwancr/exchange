
from msmbuilder.geometry import asa
from msmbuilder import Trajectory
from msmbuilder import io
import numpy as np

p=Trajectory.load_from_pdb('barstarH.pdb')

sasa = asa.calculate_asa(p, n_sphere_points=150)

io.saveh('sasa.h5', sasa=sasa)
