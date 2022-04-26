from src.environments.power_voltage_env import PowerVoltageEnv
from src.devices.ev_charger import EVChargerDevice
from src.devices.feeder import FeederDevice
from src.devices.load import LoadDevice
from src.devices.pv import PVDevice
import numpy as np
import random


def create_iee16_grid(config, ps_samplers_dict, ps_metadata, canopy_sampler, canopy_metadata,
                      price_sampler, price_metadata, ev_sampler, elaadnl_metadata, basic_arrival_rate):
    # IEEE16 topology
    feeder_inds = [0, 1, 2]
    pv_inds = [16, 17, 18, 19, 20, 21]
    load_inds = []
    ev_charger_inds = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    connections = [(0, 3), (3, 4), (3, 5), (5, 6), (1, 7), (7, 8), (7, 9), (8, 10), (8, 11),
                   (2, 12), (12, 13), (12, 14), (14, 15), (5, 16), (4, 17), (8, 18), (9, 19), (11, 20),
                   (14, 21), (4, 10), (9, 13), (6, 15)]
    n_devices = len(feeder_inds) + len(ev_charger_inds) + len(pv_inds) + len(load_inds)
    # Assign Pecan Street PV to each PV device
    houses_with_pv = [hid for hid in ps_metadata if ps_metadata[hid]['has_pv']]
    hids_for_pvs = [random.choice(houses_with_pv) for _ in range(len(pv_inds))]
    # Assign Pecan Street loads to each load device
    hids_for_loads = [random.choice(list(ps_metadata.keys())) for _ in range(len(load_inds))]
    # Create devices
    devices = [None for _ in range(n_devices)]
    feeders = []
    for f_ind in feeder_inds:
        feeders.append(FeederDevice('Feeder%d' % f_ind, config['t0_hr'], config['dt_min'], price_sampler,
                                    p_internal_min=config['feeder_p_min']))
        devices[f_ind] = feeders[-1]
    pvs = []
    for ind, pv_ind in enumerate(pv_inds):
        hid = hids_for_pvs[ind]
        pvs.append(PVDevice('PV_%s' % pv_ind, config['t0_hr'], config['dt_min'], ps_samplers_dict[hid], 0,
                            p_internal_min=-config['ps_pvs_rated_power']))
        devices[pv_ind] = pvs[-1]
    loads = []
    for ind, load_ind in enumerate(load_inds):
        hid = hids_for_loads[ind]
        loads.append(LoadDevice('Load_%s' % load_ind,  config['t0_hr'], config['dt_min'], ps_samplers_dict[hid]))
        devices[load_ind] = loads[-1]
    ev_chargers = []
    for ev_charger_ind in ev_charger_inds:
        ev_chargers.append(EVChargerDevice('EVCharger%d' % ev_charger_ind, config['t0_hr'], config['dt_min'],
                                           config['ev_dt_min'], ev_sampler,
                                           basic_arrival_rate * config['avg_evs_per_day']))
        devices[ev_charger_ind] = ev_chargers[-1]

    # Create topology
    conductance_matrix = np.zeros((n_devices, n_devices))
    i_max_matrix = np.zeros((n_devices, n_devices))
    for i_from, i_to in connections:
        conductance_matrix[i_from, i_to] = config['g']
        conductance_matrix[i_to, i_from] = config['g']
        i_max_matrix[i_from, i_to] = config['i_max']
        i_max_matrix[i_to, i_from] = config['i_max']
    # Create env
    env = PowerVoltageEnv(devices, conductance_matrix, i_max_matrix, config)
    return env
