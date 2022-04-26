import pandas as pd

from src.timedata_util import create_timesteps_hr, t_str_to_t_hr, t_hr_to_t_str
from src.samplers.ev_session_sampler import EVSessionSampler
from src.devices.device import Device
from src.devices.ev import EV
import numpy as np


class EVChargerDevice(Device):

    def __init__(self,
                 name: str,
                 t0_hr: float,
                 dt_min: int,
                 ev_dt_min: int,
                 sampler: EVSessionSampler,
                 arrival_rate: np.array,
                 utility_coef: float = 0,
                 v_internal_min: float = 300,
                 v_internal_max: float = 400,
                 p_internal_min: float = 0,
                 p_internal_max: float = 10):
        """ This class now only samples future deterministically. Not important for RL. """
        super().__init__(name, t0_hr, dt_min, sampler, utility_coef, v_internal_min,
                         v_internal_max, p_internal_min, p_internal_max)
        self.type = 'ev_charger'
        self.ev_timesteps_hr = create_timesteps_hr(t0_hr, ev_dt_min)
        self.ev_dt_min = ev_dt_min
        self.arrival_rate = arrival_rate

    def reset(self,  seed=None):
        ev_sessions = self.sampler.sample_day_evs(self.arrival_rate, seed)
        day_data = [None for _ in range(len(self.timesteps_hr))]
        evs_dict = {}
        for ev_ind, (t_arr_hr, t_dep_hr, p_demand, utility_coef) in enumerate(ev_sessions):
            ev = EV('EV_at_%s' % self, 0, p_demand, p_demand, t_arr_hr, t_dep_hr, utility_coef)
            ev.reset()
            evs_dict[ev_ind] = ev
            t_arr_ind = self.timesteps_hr.index(t_arr_hr)
            t_dep_ind = self.timesteps_hr.index(t_dep_hr)
            for ti in range(t_arr_ind, t_dep_ind):
                day_data[ti] = int(ev_ind)

        self.info['current_episode_sampled'] = True
        self.info['current_episode_evs_dict'] = evs_dict
        self.info['current_episode_data'] = pd.Series(day_data, name='EV_ind',
                                                      index=[t_hr_to_t_str(t_hr) for t_hr in self.timesteps_hr])
        self.info['current_episode_power'] = []
        self.info['current_episode_voltage'] = []
        self.update_timestep(self.t0_str)

    def update_timestep(self, t_str):
        assert self.info['current_episode_sampled'], 'Device %s tries to update_params without episode sampled' % self
        ev_ind = self.info['current_episode_data'][t_str]

        ev = self.info['current_episode_evs_dict'][ev_ind] if ev_ind is not None and not np.isnan(ev_ind) else None
        self.info['current_t_str'] = t_str
        self.info['current_t_hr'] = t_str_to_t_hr(t_str)
        self.info['active_ev'] = ev
        if ev is None:
            self.p_min, self.p_max = 0, 0
            self.utility_coef = 0
        else:
            self.p_max = min((ev.free_space / ev.charge_coef) / (self.dt_min / 60), self.p_internal_max)
            self.p_min = max(-(ev.current_soc / ev.charge_coef) / (self.dt_min / 60), self.p_internal_min)
            self.utility_coef = ev.utility_coef

    def update_power_and_voltage(self, p, v, target_dt_min=None, method=np.mean):
        assert self.p_min - 1e-5 <= p <= self.p_max + 1e-5, \
            'Device %s received p which is out of bounds: %.2f' % (self, p)
        assert self.v_min - 1e-5 <= v <= self.v_max + 1e-5, \
            'Device %s received v which is out of bounds: %.2f' % (self, v)
        p = min(max(self.p_min, p), self.p_max)
        v = min(max(self.v_min, v), self.v_max)
        reward = 0
        if self.info['active_ev'] is not None:
            reward = self.info['active_ev'].charge(p * self.dt_min / 60)
        self.info['current_episode_power'].append(p)
        self.info['current_episode_voltage'].append(v)
        return reward

    def get_utility_coef(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        ev_ind = self.info['current_episode_data'][t_str]
        if ev_ind is None or np.isnan(ev_ind):
            return [0]
        else:
            return [self.info['current_episode_evs_dict'][ev_ind].utility_coef]

    def get_p_bounds(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        ev_ind = self.info['current_episode_data'][t_str]
        if ev_ind is None or np.isnan(ev_ind):
            return [0], [0]
        else:
            return [self.p_internal_min], [self.p_internal_max]

    def get_v_bounds(self, t_str, target_dt_min=None, uncertainty='deterministic'):
        return [self.v_min], [self.v_max]

