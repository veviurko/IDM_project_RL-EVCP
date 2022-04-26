from src.environments.example_grids.ieee16grid import create_iee16_grid
from src.timedata_util import t_min_to_t_str, t_str_to_t_hr
import numpy as np


def get_arrival_rate(ev_arrivals_per_minute, ev_timesteps_hr, ev_dt_min):
    rescaling_map = {t_min_to_t_str(i * ev_dt_min + j): t_min_to_t_str(i * ev_dt_min)
                         for i in range(int(24 * 60 / ev_dt_min)) for j in range(ev_dt_min)}
    basic_arrival_rate = np.empty(len(ev_timesteps_hr))
    for t_str, count in ev_arrivals_per_minute.items():
        t_hr = t_str_to_t_hr(rescaling_map[t_str])
        basic_arrival_rate[ev_timesteps_hr.index(t_hr)] += count
    return basic_arrival_rate / sum(basic_arrival_rate)


SUPPORTED_GRIDS = ['ieee16']


def create_env(config, ps_samplers_dict, ps_metadata, canopy_sampler, canopy_metadata,
               price_sampler, price_metadata, ev_sampler, elaadnl_metadata):
    # Update samplers (resamples data id dt was changed)\
    dt_min = config['dt_min']
    for sampler in ps_samplers_dict.values():
        if dt_min != sampler.dt_min:
            sampler.setup_dt(dt_min)
    if canopy_sampler.dt_min != dt_min:
        canopy_sampler.setup_dt(dt_min)
    if price_sampler.dt_min != dt_min:
        price_sampler.setup_dt(dt_min)
    if ev_sampler.dt_min != dt_min:
        ev_sampler.setup_dt(dt_min)

    grid_to_use = config['grid_to_use']
    basic_arrival_rate = get_arrival_rate(elaadnl_metadata['arrival_counts'], ev_sampler.ev_timesteps_hr,
                                          ev_sampler.ev_dt_min)
    assert grid_to_use in SUPPORTED_GRIDS, 'Grid %s is not supported. Choose a grid from the list:\n %s' % \
                                           (grid_to_use,  SUPPORTED_GRIDS)
    if grid_to_use == 'ieee16':
        env = create_iee16_grid(config, ps_samplers_dict, ps_metadata, canopy_sampler, canopy_metadata,
                                price_sampler, price_metadata, ev_sampler, elaadnl_metadata, basic_arrival_rate)
    return env
