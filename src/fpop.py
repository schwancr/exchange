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
        timepoints, exchange_prob_f=None, interp_rates=True, init_pops=None, lagtime=1.):
        """
        initialize the exchanger

        Parameters
        ----------
        tProb : scipy.sparse matrix
            transition probability matrix for your MSM
        avg_sasas : np.ndarray
            sasas for the sidechains you want to simulate FPOP for. Rows should
            be the states, while columns should correspond to the residues of interest
        protein_conc : float
            initial protein concentration in M
        OH_conc : float
            initial OH concentration upon excitation in M
        pdb : msmbuilder.Trajectory
            pdb to find the residue names
        timepoints : np.ndarray or list
            timepoints to simulate an fpop experiment for. Should be in FRAMES
        exchange_prob_f : function
            function that takes a state_var for each state and returns a
            probability of exchanging in one lagtime
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
        
        self.res_rate_ex = []
        self.res_names = []
        for i in np.unique(pdb['ResidueID']):
            res_name = pdb['ResidueNames'][np.where(pdb['ResidueID'] == i)][0].lower()
            if not res_name in self.rate_dict.keys():
                raise Exception("unrecognized residue %s" % self.residue)

            self.res_names.append(res_name)
            self.res_rate_ex.append(self.rate_dict[res_name])

        self.num_res = len(self.res_rate_ex)
        self.res_rate_ex = np.array(self.res_rate_ex)
        self.res_rate_ex_per_state = self._get_state_rates()

        if init_pops is None:
            self.last_populations = np.ones((self.num_states, self.num_res)) / float(self.num_states)

        else:   
            self.last_populations = np.vstack([init_pops] * self.num_res).T

        self.t = 0
    
        self.timepoints = np.unique(timepoints)  # unique checks for repeats and also sorts

        self.exchanged_per_res = np.zeros((1, self.num_res))

    def _get_state_rates(self):

        ref_sasa_dict = {
        'ala' : 0.67,
        'arg' : 1.96,
        'asn' : 1.13,
        'asp' : 1.06,
        'cys' : 1.04,
        'gln' : 1.44,
        'glu' : 1.38,
        'gly' :   -1,  # label to remove later
        'his' : 1.51,
        'ile' : 1.40,
        'leu' : 1.37,
        'lys' : 1.67,
        'met' : 1.60,
        'phe' : 1.75,
        'pro' : 1.05,
        'ser' : 0.80,
        'thr' : 1.02,
        'trp' : 2.17,
        'tyr' : 1.87,
        'val' : 1.17
        }

        ref_sasa_per_res = []
        for name in self.res_names:
            ref_sasa_per_res.append(ref_sasa_dict[name])

        ref_sasa_per_res = np.array(ref_sasa_per_res)

        perc_sasa = self.avg_sasas / np.reshape(ref_sasa_per_res, (1, -1))

        perc_sasa[:, np.where(ref_sasa_per_res < 0)[0]] = 0
        # for glycine, need to set to zero

        rate_mat = perc_sasa ** 10 * self.res_rate_ex
        return rate_mat

        # first just do On or Off
        #temp_rate_mat = np.vstack([self.res_rate_ex] * self.num_states)

        #temp_rate_mat[np.where(self.avg_sasas < 0.5)] = 0

        #return temp_rate_mat

    def _derivX(self, t, X):
        # first num_states are the state populations
        # last piece of X is the OH concentration
        
        dX = np.zeros(self.num_states * self.num_res + 1)

        M = X[:-1].reshape((self.num_states, self.num_res))
        OH = X[-1]

        temp_mat = self.res_rate_ex_per_state * (self.last_populations * self.init_protein_conc - M)
        dLij_dt = temp_mat * OH

        dOH_dt = - np.sum(temp_mat) * OH - self.rate_quench * (self.quench_conc - self.OH_conc + OH + np.sum(M)) * OH
        
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
        
        #print self.last_populations.flatten()
        #X = scipy.integrate.odeint(self._derivX, 
        #        np.concatenate((self.last_populations.flatten(), [self.OH_conc])), 
        #        np.linspace(0, self.lagtime, 30)[1:])[-1]  # only care about end result
    
        solver = scipy.integrate.ode(self._derivX)
        solver.set_integrator('vode', method='bdf')
        solver.set_initial_value(np.concatenate((np.zeros(self.num_states * self.num_res), [self.OH_conc])))

        #solver.integrate(solver.t + self.lagtime)

        m = 10
        for i in range(1, m + 1):
            solver.integrate(solver.t + self.lagtime / float(m) * i)
            
        X = solver.y
        final_OH_conc = X[-1]

        total_exchanged = np.reshape(X[:-1], (self.num_states, self.num_res))

        self.quench_conc = self.quench_conc - self.OH_conc + final_OH_conc + np.sum(total_exchanged)
        self.OH_conc = final_OH_conc
    
        #print self.last_populations.sum(axis=0)
        #print total_exchanged
        #print final_OH_conc, self.quench_conc
        exchange_probs = total_exchanged / (self.last_populations * self.init_protein_conc)

        return exchange_probs


    def next_step(self):
        """
        simulate one more step of the reaction
        """
        
        exchange_probs = self.get_exchange_probs()

        for i in xrange(self.num_res):
            X = scipy.sparse.dia_matrix((np.reshape((1 - exchange_probs[:, i]), (1, -1)), np.array([0])), shape=(self.num_states, self.num_states))
            self.last_populations[:, i] = X.dot(self.tProb).T.dot(self.last_populations[:, i])

        self.exchanged_per_res = np.vstack([self.exchanged_per_res, 1 - self.last_populations.sum(axis=0)])

        self.t += self.lagtime


    def _rewind(self, t, populations):
        """
        rewind the simulation to some time t and reset OH and quencher conc
        
        Parameters:
        -----------
        t : float
            time to rewind to (in FRAMES)
        populations : np.ndarray
            populations corresponding to "last_populations" at time t0
        """

        if self.last_populations.shape != populations.shape:
            raise Exception("invalid input for 'populations'")

        num_frames = int(t)

        self.exchanged_per_res = self.exchanged_per_res[:num_frames]

        self.last_populations = populations

        self.t = t * self.lagtime

        self.OH_conc = float(self.init_OH_conc)
        self.quench_conc = float(self.init_quench_conc)


    def _compute_total_products(self, num_labels):
        """
        compute the product distribution or peptides with 0, 1, 2, .., N labels
        
        Parameters
        ----------
        num_labels : int
            number of labels to compute populations of
        
        Returns:
        --------
        product_dist : np.ndarray

            product distribution of each multi-labeled product 0 ... N
        """

        product_dist = np.ones(num_labels + 1) * -1  # add one to include 0-label

        res_probs = self.exchanged_per_res[-1]
        not_res_probs = 1 - res_probs
        ratio_probs = res_probs / not_res_probs
        ratio_probs_sum = ratio_probs.sum()

        product_dist[0] = np.product(not_res_probs)

        for i in xrange(1, num_labels + 1):
            product_dist[i] = ratio_probs_sum * product_dist[i - 1]

        return product_dist


    def run(self, return_res_pops=False, num_labels=5, max_steps=10000, tol=1E-6):
        """
        Run the simulation!!
        This will calculate the total exchanged as a function of the timepoints
        at which the OH radical was produced.
        
        Parameters
        ----------
        return_res_pops : bool, optional
            return both the total products formed as well as the individual
            populations per residue (default: False)
        num_labels : int, optional
            number of products to calculate probabilities for. (default: 5)
        max_steps : int, optional
            maximum number of steps to take when trying to compute converged
            exchange populations (default: 10000)
        tol : float, optional
            tolerance to use for saying the exchanged populations have converged
            (default: 1E-6)

        Returns:
        --------
        product_dist : np.ndarray
            distribution of products, up to N labels
        exchange_per_res : np.ndarray
            amount exchanged per residue as a function of time. rows correspond
            to the timepoints in 'timepoints' and columns are residues
        """

        temp_populations = None  # hold the populations from the frame right before
            # exchanging so we can use it in the next time point without redoing
            # some propagating
        
        total_exchanged_per_res = np.ones((self.timepoints.shape[0], self.num_res)) * -1
        total_product_pops = np.ones((self.timepoints.shape[0], num_labels + 1)) * -1

        buff = 0
        for i, n0 in enumerate(self.timepoints):
            
            if i > 0:
                print temp_populations
                self._rewind(self.timepoints[i - 1], temp_populations)
                buff = self.timepoints[i - 1]

            self.t0 = n0 * self.lagtime
            print i
            for frame in xrange(max_steps):

                if (frame + buff) == n0:
                    print 'here'
                    temp_populations = self.last_populations  # we will exchange this step

                self.next_step()

                if (len(self.exchanged_per_res) > 1) and ((frame + buff) > n0):
                    if np.abs(self.exchanged_per_res[-1] - self.exchanged_per_res[-2]).max() < tol:
                        print "broke"
                        break

            total_exchanged_per_res[i] = self.exchanged_per_res[-1]
    
            total_product_pops[i] = self._compute_total_products(num_labels)

        if return_res_pops:
            return total_product_pops, total_exchanged_per_res

        return total_product_pops
