from src.timedata_util import shift_date, t_min_to_t_str, t_hr_to_t_str, create_timesteps_hr, round_t_hr
from typing import List, Dict
import pandas as pd
import numpy as np


class EVSessionSampler:

    def __init__(self,
                 t0_hr: float,
                 dt_min: int,
                 ev_dt_min: int,
                 sampling_dt_min: int,
                 path_to_dataset: str,
                 utility_coef_mean: float,
                 utility_coef_scale: float,
                 apply_gaussian_noise: bool = True,
                 noise_scale: float = .1,
                 ):

        self.path_to_dataset = path_to_dataset

        self.t0_hr = t0_hr
        self.t0_str = t_hr_to_t_str(t0_hr)

        self.setup_dt(dt_min)

        self.ev_dt_min = ev_dt_min
        self.ev_timesteps_hr = create_timesteps_hr(t0_hr, ev_dt_min)
        self.ev_timesteps_str = [t_hr_to_t_str(t_hr) for t_hr in self.ev_timesteps_hr]

        self.sampling_dt_min = sampling_dt_min

        self.data = self._load_and_rescale_time()

        self.months = self.data['month'].unique()

        self.utility_coef_mean = utility_coef_mean
        self.utility_coef_scale = utility_coef_scale

        self.apply_gaussian_noise = apply_gaussian_noise
        self.noise_scale = noise_scale

    def _load_and_rescale_time(self):
        data = pd.read_csv(self.path_to_dataset)
        rescaling_map = {t_min_to_t_str(i * self.sampling_dt_min + j): t_min_to_t_str(i * self.sampling_dt_min)
                         for i in range(int(24 * 60 / self.sampling_dt_min)) for j in range(self.sampling_dt_min)}
        data['time_start'] = data['time_start'].apply(lambda x: rescaling_map[x])
        data['time_end'] = data['time_end'].apply(lambda x: rescaling_map[x])
        data['month'] = data['date_start'].apply(lambda x: x[5:7])
        return data

    def _sample_utility_coef(self):
        return np.random.normal(loc=self.utility_coef_mean, scale=self.utility_coef_scale)

    def setup_dt(self, dt_min):
        self.dt_min = dt_min
        self.timesteps_hr = create_timesteps_hr(self.t0_hr, dt_min)
        self.timesteps_str = [t_hr_to_t_str(t_hr) for t_hr in self.timesteps_hr]

    def sample_day_evs(self, arrival_rate, seed=None):
        assert len(arrival_rate) == len(self.ev_timesteps_hr), 'Arrival rate and timesteps must have same length!'
        node_is_busy = np.zeros_like(self.ev_timesteps_hr, dtype='bool')
        evs = []
        for t_arr_ev_ind, t_arr_hr in enumerate(self.ev_timesteps_hr[:-1]):
            if node_is_busy[t_arr_ev_ind]:
                continue
            if np.random.uniform(0, 1) <= arrival_rate[t_arr_ev_ind]:
                t_dep_hr, p_demand, utility_coef = self.sample_ev_session(t_arr_hr)
                t_dep_ev_ind = self.ev_timesteps_hr.index(t_dep_hr)
                node_is_busy[t_arr_ev_ind: t_dep_ev_ind + 1] = True
                evs.append((t_arr_hr,  t_dep_hr, p_demand, utility_coef))
        return evs

    def sample_ev_session(self, t_hr):
        t_sample_str = t_hr_to_t_str(round_t_hr(t_hr, self.sampling_dt_min))
        fitting_sessions = self.data[self.data['time_start'].isin([t_sample_str])].reset_index(drop=True)
        session = fitting_sessions.iloc[np.random.randint(len(fitting_sessions))]
        p_demand, t_charge_hr = session[['charged', 'duration']].values
        if self.apply_gaussian_noise:
            p_demand *= np.random.normal(loc=1, scale=self.noise_scale)
            t_charge_hr *= np.random.normal(loc=1, scale=self.noise_scale)
        max_t_charge_hr = 24 - (t_hr - self.t0_hr) if t_hr >= self.t0_hr else self.t0_hr - t_hr
        t_charge_hr = max(min(max_t_charge_hr - 1e-6, t_charge_hr), self.ev_dt_min / 60)
        t_dep_hr = round_t_hr((t_hr + t_charge_hr) % 24, self.ev_dt_min)
        utility_coef = self._sample_utility_coef()
        return t_dep_hr, p_demand, utility_coef
