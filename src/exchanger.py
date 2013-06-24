import scipy.sparse
import numpy as np

class BaseExchanger(object):
    """
    base class for exchanger objects that model exchange type reactions
    """
    def __init__(self, tProb, state_var, exchange_prob_f, init_pops=None,
        lagtime=1., exchange_is_constant=False):
        """
        initialize the exchanger

        Parameters
        ----------
        tProb : scipy.sparse matrix
            transition probability matrix for your MSM
        state_var : np.ndarray
            variable defining the state's ability to exchange
        exchange_prob_f : function
            function that takes a state_var for each state and returns a
            probability of exchanging in one lagtime
        init_pops : np.ndarray, optional
            initial populations for the simulation. If None, then use equal
            population in all states
        lagtime : float, optional
            lagtime of the MSM
        exchange_is_constant : bool, optional
            if the exchange probability of a state is constant over time
            then pass True (default: False)
        """
    
        self.init_tProb = tProb
        self.num_states = tProb.shape[0]
    
        self.exchange_prob_f = exchange_prob_f
        #self._construct_exchange_tProb()
        self.lagtime = lagtime
        self.t = 0

        exchange_is_constant = exchange_is_constant


    def _construct_exchange_tProb(self):
        """
        construct the block transition matrix that includes the exchange
        reaction.
        """

        self.exchange_probs = self.exchange_prob_f(state_vars)

        self.tProb = scipy.sparse.zeros((self.num_states * 2, self.num_states * 2))

        self.tProb[:self.num_states, :self.num_states] = self.init_tProb * np.reshape(1 - exchange_probs, (-1, 1))
        self.tProb[:self.num_states, self.num_states:] = self.init_tProb * np.reshape(exchange_probs, (-1, 1))
        self.tProb[self.num_states:, self.num_states:] = self.init_tProb 

        # We assume above that the transition matrix is a block matrix:
        # 
        #  [ T00 T01 ]
        #  [ T10 T11 ]
        # 
        # T00 = p(no exchange) * T
        # T01 = p(exchange) * T
        # T10 = 0
        # T11 = T
        # 
        # NOTE: This means the new transition matrix is no longer reversible!!!
        # that's ok, and we could fudge it and make it super slow if we need reversibility
        # these exchange reactions are usually thought to be irreversible anyway


    def next(self):
        """
        simulate one more step of the reaction
        """
        if not self.exchange_is_constant:
            self._construct_exchange_tProb()

        next_pops = (self.tProb.T.dot(self.populations[-1])).T

        self.populations = np.concatenate((self.populations, next_pops))
        
        self.t += self.lagtime


    def simulate(self, num_steps):
        """
        simulate step(s) of the reaction
        
        Parameters
        ----------
        num_steps : int
            number of steps to simulate
        """
        
        for i in xrange(num_steps):
            self.next()

