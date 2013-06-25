from exchanger import BaseExchanger
import scipy.sparse
import numpy as np
from scipy.integrate import ode
import copy

RATE_CONSTANTS = {
'cys' : 3.4E10,
'tyr' : 1.3E10,
'his' : 1.3E10,
'met' : 8.3E9,
'phe' : 6.5E9,
'arg' : 3.5E9, # note this may not be a plus n*16 since it often breaks off the nitrogen group on the end
'leu' : 1.7E9,
'ile' : 0, # Not available in paper
'trp' : 1.3E9,
'pro' : 0, # not available in paper
'val' : 7.6E8,
'thr' : 5.1E8,
'ser' : 3.2E8,
'glu' : 2.3E8,
'gln' : 0, # not available in paper
'ala' : 7.7E7,
'asp' : 7.5E7,
'asn' : 4.9E7,
'lys' : 3.5E7,
'gly' : 1.7E7
}

# ^^^^ These rate constants are from Xu et al. (2003) Anal. Chem.
#  (DOI: 10.1021/ac035104h)

RATE_CONSTANTS_INTERP = copy.copy(RATE_CONSTANTS)
RATE_CONSTANTS_INTERP['ile'] = 1E9
RATE_CONSTANTS_INTERP['pro'] = 1E8
RATE_CONSTANTS_INTERP['gln'] = 1E8
# ^^^ Order of magnitude interpolation based on my intuition
# ALL RATE CONSTANTS ARE IN M^-1 s^-1

class FPOPExchanger(object):
    """
    This simulator is slightly different from the BaseExchanger
    The simulation works as follows:
    1) Each timepoint is collected by a separate simulation
    2) exchange probability is zero until some t0, according to
        when the laser photolyzes the H2O2
    3) at this point exchange can occur, and the simulation
        is run until the exchange probability decreases below
        prob_threshold
    4) then the total amount exchanged is used for computing
        the total exchanged at that timepoint

    """
    
    def __init__(self, tProb, avg_sasas, protein_conc, OH_conc, quench_conc, pdb,
        exchange_prob_f=None, interp_rates=True, init_pops=None, lagtime=1.):
        """
        initialize the exchanger

        Parameters
        ----------
        tProb : scipy.sparse matrix
            transition probability matrix for your MSM
        avg_sasas : np.ndarray
            sasas for the sidechains you want to simulate FPOP for. Rows should
            be the states, while columns should correspond to the residues of interest
        exchange_prob_f : function
            function that takes a state_var for each state and returns a
            probability of exchanging in one lagtime
        protein_conc : float
            initial protein concentration in M
        OH_conc : float
            initial OH concentration upon excitation in M
        residue : str
            three letter residue name abbreviation (e.g. gly, thr, trp, etc.)
        interp_rates : bool, optional
            the rate constants are unavailable for ILE, PRO, and GLN so
            Christian Schwantes guessed an order of magnitude value for each
            based on interpolating with chemical intuition. If you pass false,
            then those residues will NEVER exchange (default: True)
        init_pops : np.ndarray, optional
            initial populations for the simulation. If None, then use equal
            population in all states
        lagtime : float, optional
            lagtime of the MSM in SECONDS!!!!
        """

        self.tProb = tProb
        self.num_states = tProb.shape[0]
    
        self.avg_sasas = avg_sasas

        self.lagtime = lagtime

        self.protein_conc = protein_conc
        self.init_protein_conc = protein_conc
        self.init_OH_conc = OH_conc
        self.OH_conc = OH_conc
        self.init_quench_conc = quench_conc
        self.quench_conc = quench_conc
        
        if interp_rates:
            self.rate_dict = RATE_CONSTANTS_INTERP
            self.rate_quench = self.rate_dict['gln']
        else:
            self.rate_dict = RATE_CONSTANTS
            self.rate_quench = 1E8  # guess at the quench rate..
        
        #self._setup_ODE()
        
        self.res_rate_ex = []
        for i in np.unique(pdb['ResidueID']):
            res_name = pdb['ResidueNames'][np.where(pdb['ResidueID'] == i)][0].lower()
            if not res_name in self.rate_dict.keys():
                raise Exception("unrecognized residue %s" % self.residue)

            self.res_rate_ex.append(self.rate_dict[res_name])

        self.num_res = len(self.res_rate_ex)
        self.res_rate_ex = np.array(self.res_rate_ex)
        self.res_rate_ex_per_state = self._get_state_rates()

        if init_pops is None:
            self.last_populations = np.ones((self.num_states, self.num_res)) / float(self.num_states)

        else:   
            self.last_populations = np.vstack([init_pops] * self.num_res).T

        self.t = 0
    
        self.t0 = 1E-6

        self.exchanged_per_res = np.zeros((1, self.num_res))

    def _get_state_rates(self):

        # first just do On or Off
        temp_rate_mat = np.vstack([self.res_rate_ex] * self.num_states)

        temp_rate_mat[np.where(self.avg_sasas < 0.3)] = 0

        return temp_rate_mat

    def _derivX(self, X, t):
        # first num_states are the state populations
        # last piece of X is the OH concentration
        
        dX = np.zeros(self.num_states * self.num_res + 1)

        M = X[:-1].reshape((self.num_states, self.num_res))
        OH = X[-1]

        temp_mat = self.res_rate_ex_per_state * ((self.last_populations[:, :]) * self.init_protein_conc - M)
        dLij_dt = temp_mat * OH

        dOH_dt = - np.sum(temp_mat) * OH - self.rate_quench * (self.quench_conc - self.OH_conc + OH + np.sum(M))
        
        dX[:-1] = dLij_dt.flatten()
        dX[-1] = dOH_dt

        return dX

    
    def _jacX(self, X, t):

        # This jacobian is too big... probably can't use it
 
        raise Exception("jacobian is too big... (num_states * num_res + 1)**2")

    def get_exchange_probs(self):
        """
        compute the probability of exchanging in the next lagtime seconds
        """

        if self.t < self.t0:
            return np.zeros((self.num_states, self.num_res))
        
        X = scipy.integrate.odeint(self._derivX, 
                np.concatenate((self.last_populations.flatten(), [self.OH_conc])), 
                50E-9)[-1]  # only care about end result

        print X.shape
        final_OH_conc = X[-1]
        total_exchanged = np.reshape(X[:-1], (self.num_states, self.num_res))
        print total_exchanged

        self.quench_conc = self.quench_conc - self.OH_conc + final_OH_conc + np.sum(total_exchanged)
        self.OH_conc = final_OH_conc
    
        exchange_probs = total_exchanged / (self.last_populations * self.protein_conc)

        return exchange_probs


    def next_step(self):
        """
        simulate one more step of the reaction
        """
        
        exchange_probs = self.get_exchange_probs()

        for i in xrange(self.num_res):
            X = scipy.sparse.eye(self.num_states, self.num_states) * (1 - exchange_probs[:, i])

            self.last_populations[:, i] = X.dot(self.tProb.T.dot(self.last_populations[:, i]))

        self.exchanged_per_res = np.vstack([self.exchanged_per_res, 1 - self.last_populations.sum(axis=0)])

        self.t += self.lagtime
