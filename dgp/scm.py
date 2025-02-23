import numpy as np
from scipy import stats


class SimpleSCM:
    
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self.u_x = stats.truncnorm(0, np.infty, scale=5)
        self.u_y = stats.norm(scale=2)
        self.u_z = stats.truncnorm(0, np.infty, scale=5)
        self.u_w = stats.gamma(a=1.7, scale=1)
        
    def sample(self, sample_size=100, treatment_value=None):
        """Samples from the SCM"""
        if self.random_seed:
            np.random.seed(self.random_seed)
        
        u_x = self.u_x.rvs(sample_size)
        u_y = self.u_y.rvs(sample_size)
        u_z = self.u_z.rvs(sample_size)
        u_w = self.u_w.rvs(sample_size)
        
        if treatment_value:
            x = np.array([treatment_value]*sample_size)
        else:
            x = u_x + u_z
          
        z = u_z
        y = -2*x + 8*z + 0.5*u_y
        w = -2*x - 18*y + 0.2*u_w
        
        return x, y, z, w
    
    def intervene(self, treatment_value, sample_size=100):
        """Intervenes on the SCM"""
        return self.sample(treatment_value=treatment_value, sample_size=sample_size)