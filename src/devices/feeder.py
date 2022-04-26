from src.timedata_util import t_hr_to_t_str, t_str_to_t_hr
from src.samplers.time_series_sampler import TimeSeriesSampler
from src.devices.device import Device
import numpy as np


class FeederDevice(Device):

    def __init__(self,
                 name: str,
                 t0_hr: float,
                 dt_min: int,
                 sampler: TimeSeriesSampler,
                 utility_coef: float = 0,
                 v_internal_min: float = 300,
                 v_internal_max: float = 400,
                 p_internal_min: float = -10,
                 p_internal_max: float = 0):

        super().__init__(name, t0_hr, dt_min, sampler, utility_coef, v_internal_min,
                         v_internal_max, p_internal_min, p_internal_max)
        self.type = 'feeder'
        self.allowed_uncertainties = ['deterministic', 'monthly scenarios', 'monthly average']

    def reset(self, date):
        day_data = self.sampler.sample_day_data(date, self.type)
        self.info['current_episode_sampled'] = True
        self.info['current_episode_date'] = date
        self.info['current_episode_data'] = day_data
        self.info['current_episode_month_data'] = self.sampler.sample_month_data(date, self.type, average=False)
        self.info['current_episode_power'] = []
        self.info['current_episode_voltage'] = []
        self.update_timestep(self.t0_str)

    def update_timestep(self, t_str):
        assert self.info['current_episode_sampled'], 'Device %s tries to update_params without episode sampled' % self
        self.utility_coef = self.info['current_episode_data'][t_str]
        self.info['current_t_str'] = t_str
        self.info['current_t_hr'] = t_str_to_t_hr(t_str)

    def update_power_and_voltage(self, p, v):
        assert self.p_min - 1e-5 <= p <= self.p_max + 1e-5, \
            'Device %s received p which is out of bounds: %.2f' % (self, p)
        assert self.v_min - 1e-5 <= v <= self.v_max + 1e-5, \
            'Device %s received v which is out of bounds: %.2f' % (self, v)
        p = min(max(self.p_min, p), self.p_max)
        v = min(max(self.v_min, v), self.v_max)
        r = self.utility_coef * p * self.dt_min / 60
        self.info['current_episode_power'].append(p)
        self.info['current_episode_voltage'].append(v)
        return r

    def get_p_bounds(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        assert uncertainty in self.allowed_uncertainties, 'Unknown uncertainty %s' % uncertainty
        return [self.p_min], [self.p_max]

    def get_v_bounds(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        return [self.v_min], [self.v_max]

    def get_utility_coef(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        assert uncertainty in self.allowed_uncertainties, 'Unknown uncertainty %s' % uncertainty
        if target_dt_min is None:
            target_dt_min = self.dt_min
        t_hr = t_str_to_t_hr(t_str)
        t_hr_list = [(t_hr + i * self.dt_min / 60) % 24 for i in range(target_dt_min // self.dt_min)]
        t_str_list = [t_hr_to_t_str(t_hr) for t_hr in t_hr_list]
        if uncertainty == 'monthly scenarios':
            utility_coefs_t = [np.mean([day_data[t_str] for t_str in t_str_list])
                               for day_data in self.info['current_episode_month_data']]
        elif uncertainty == 'monthly average':
            utility_coefs_t = [sum([np.mean([day_data[t_str] for t_str in t_str_list])
                               for day_data in self.info['current_episode_month_data']]) /
                               len(self.info['current_episode_month_data'])]
        else:
            utility_coefs_t = [np.mean([self.info['current_episode_data'][t_str] for t_str in t_str_list])]
        return utility_coefs_t


