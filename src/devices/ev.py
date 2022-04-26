


class EV:

    def __init__(self,
                 name: str,
                 soc_arr: float,
                 soc_goal: float,
                 soc_max: float,
                 t_arr_hr: float,
                 t_dep_hr: float,
                 utility_coef: float,
                 charge_coef: float = 1.,
                 ):

        self.name = name
        self.soc_arr = soc_arr
        self.soc_goal = soc_goal
        self.soc_max = soc_max

        self.t_arr_hr = t_arr_hr
        self.t_dep_hr = t_dep_hr

        self.utility_coef = utility_coef
        self.charge_coef = charge_coef

        self.current_soc = None

    def __repr__(self):
        return 'EV_%s_t_arr_hr=%.1f_t_dep_hr=%.1f' % (self.name, self.t_arr_hr, self.t_dep_hr)

    def __str__(self):
        return 'EV_%s_t_arr_hr=%.1f_t_dep_hr=%.1f' % (self.name, self.t_arr_hr, self.t_dep_hr)

    @property
    def free_space(self):
        if self.current_soc is None:
            return None
        else:
            return self.soc_max - self.current_soc

    def reset(self):
        self.current_soc = self.soc_arr

    def charge(self, p_kwh):
        new_soc = self.current_soc + self.charge_coef * p_kwh
        assert -1e5 <= new_soc <= self.soc_max + 1e5, 'SOC of the EV %s is getting out of bounds: %.2f' % (self,
                                                                                                           new_soc)
        new_soc = min(new_soc, self.soc_max)
        self.current_soc = new_soc
        return self.charge_coef * p_kwh * self.utility_coef
