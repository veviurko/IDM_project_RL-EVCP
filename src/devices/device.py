from src.timedata_util import t_hr_to_t_str, create_timesteps_hr


class Device:

    def __init__(self,
                 name: str,
                 t0_hr: float,
                 dt_min: int,
                 sampler,
                 utility_coef: float,
                 v_internal_min: float = 300,
                 v_internal_max: float = 400,
                 p_internal_min: float = 0,
                 p_internal_max: float = 10):
        """ Device class is used to store information and sample data for each device in the grid.
            Current implementation supports the following devices:
                'feeder' -- connection to the external grid, provides power to the grid.
                'pv'     -- solar panel, provides power to the grid. Generation capacity (p_min) is stochastic.
                'load'   -- inflexible demand that must be served (p_min = p_max). Stochastic.
                 'ev_charger' --  EV charging station. DOES NOT consider EV state-of-charge when computing p_min, p_max
                                  for future timesteps.
            Each device is creates using a sampler, that has all the data from  which the device
            parameters will be sampled.
        """
        self.name = name
        self.type = 'generic'

        self.t0_hr = t0_hr
        self.t0_str = t_hr_to_t_str(t0_hr)
        self.dt_min = dt_min

        self.timesteps_hr = create_timesteps_hr(t0_hr, dt_min)

        self.p_internal_min, self.p_internal_max = p_internal_min, p_internal_max
        self.p_min, self.p_max = float(self.p_internal_min), float(self.p_internal_max)
        self.v_internal_min, self.v_internal_max = v_internal_min, v_internal_max
        self.v_min, self.v_max = float(self.v_internal_min), float(self.v_internal_max)

        self.utility_coef = utility_coef

        self.sampler = sampler

        self.info = {'current_episode_sampled': False}

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def reset(self, date):
        """ Resets the device to the beginning of a new episode corresponding to date. """
        raise NotImplementedError

    def update_timestep(self, t_str):
        """ Updates timestep of the device by sampling p_min, p_max, v_min, v_max, u. """
        raise NotImplementedError

    def update_power_and_voltage(self, p, v):
        """ Sets up power and voltage of the device to the input values.
            Will raise error if lower/upper bound is not respected. """
        raise NotImplementedError

    def get_p_bounds(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        """ Estimates values of p_min, p_max at timestep t_str.
           target_dt_min -- used to specify a sampling timescale different from self.timesteps_hr
           uncertainty   -- how future uncertainty is resolved.  """
        raise NotImplementedError

    def get_v_bounds(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        """ Estimates values of v_min, v_max at timestep t_str.
            target_dt_min -- used to specify a sampling timescale different from self.timesteps_hr
            uncertainty   -- how future uncertainty is resolved.  """
        raise NotImplementedError

    def get_utility_coef(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        """ Estimates value of u at timestep t_str.
            target_dt_min -- used to specify a sampling timescale different from self.timesteps_hr
            uncertainty   -- how future uncertainty is resolved.  """
        raise NotImplementedError
