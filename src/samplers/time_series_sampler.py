from src.timedata_util import shift_date, t_min_to_t_str, t_hr_to_t_str, create_timesteps_hr
from typing import List, Dict
import pandas as pd
import numpy as np


class TimeSeriesSampler:

    def __init__(self,
                 t0_hr: float,
                 dt_min: int,
                 path_to_dataset: str,
                 solar_rated_power: float,
                 device_type_to_column: Dict[str, str],
                 apply_gaussian_noise: bool = True,
                 noise_scale: float = .1):

        self.path_to_dataset = path_to_dataset

        self.t0_hr = t0_hr
        self.t0_str = t_hr_to_t_str(t0_hr)
        self.dt_min = dt_min
        self.timesteps_hr = create_timesteps_hr(t0_hr, dt_min)

        self.data_raw = self._load_data()
        self.dates = self.data_raw['date_shifted'].unique()
        self.data = self.data_raw.copy()
        self.date_to_data = {}
        self.setup_dt(dt_min)

        self.solar_rated_power = solar_rated_power
        self.device_type_to_column = device_type_to_column

        self.apply_gaussian_noise = apply_gaussian_noise
        self.noise_scale = noise_scale

    def _load_data(self):
        data = pd.read_csv(self.path_to_dataset)
        data['date_shifted'] = (data['date'] + ' ' + data['time']).apply(lambda x: (shift_date(x, self.t0_hr)))
        dates_to_drop = []
        for date, count in data['date_shifted'].value_counts().items():
            if count < 24 * 60:
                dates_to_drop.append(date)
        mask_to_drop = data['date_shifted'].isin(dates_to_drop)
        data = data[~mask_to_drop].reset_index(drop=True)
        return data

    def setup_dt(self, dt_min):
        self.dt_min = dt_min
        self.timesteps_hr = create_timesteps_hr(self.t0_hr, dt_min)
        rescaling_map = {t_min_to_t_str(i * dt_min + j): t_min_to_t_str(i * dt_min)
                         for i in range(int(24 * 60 / dt_min)) for j in range(dt_min)}
        self.data = self.data_raw.copy()
        self.data['time'] = self.data_raw['time'].apply(lambda x: rescaling_map[x])
        self.data = self.data.groupby(['date', 'time']).mean().reset_index()
        self.data['date_shifted'] = (self.data['date'] + ' ' + self.data['time']).apply(lambda x:
                                                                                        (shift_date(x, self.t0_hr)))
        self.date_to_data = {date: self.data[self.data['date_shifted'].isin([date])].set_index('time', drop=True)
                             for date in self.dates}

    def sample_day_data(self, date, device_type):
        assert device_type in self.device_type_to_column, 'Wrong device type %s for sampler %s' % (device_type, self)
        data_at_date = self.date_to_data[date][self.device_type_to_column[device_type]]
        if device_type == 'pv':
            data_at_date = -np.maximum(0, data_at_date) / self.solar_rated_power
        elif device_type == 'load':
            data_at_date = np.maximum(0, data_at_date)
        if self.apply_gaussian_noise:
            data_at_date = data_at_date * np.random.normal(loc=1, scale=self.noise_scale, size=len(data_at_date))
        return data_at_date

    def sample_month_data(self, date_true, device_type, average=True):
        assert device_type in self.device_type_to_column, 'Wrong device type %s for sampler %s' % (device_type, self)
        all_dates_month = [d for d in self.dates if d[5:7] == date_true[5:7]]
        monthly_data = []
        for date in all_dates_month:
            if date == date_true:
                continue
            data_at_date = self.date_to_data[date][self.device_type_to_column[device_type]]
            if device_type == 'pv':
                data_at_date = -np.maximum(0, data_at_date) / self.solar_rated_power
            elif device_type == 'load':
                data_at_date = np.maximum(0, data_at_date)
            if self.apply_gaussian_noise:
                data_at_date = data_at_date * np.random.normal(loc=1, scale=self.noise_scale, size=len(data_at_date))
            monthly_data.append(data_at_date)
        if average:
            monthly_data = [sum(monthly_data) / len(monthly_data)]
        return monthly_data

