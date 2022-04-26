from src.timedata_util import split_dates_train_and_test_monthly
from src.samplers.time_series_sampler import TimeSeriesSampler
from src.samplers.ev_session_sampler import EVSessionSampler
import pickle


def load_samplers(config):
    # Read config
    t0_hr = config['t0_hr']
    dt_min = config['dt_min']
    ev_dt_min = config['ev_dt_min']
    ev_sampling_dt_min = config['ev_sampling_dt_min']
    path_to_data = config['path_to_data']
    days_per_month_train = config['days_per_month_train']
    ev_session_months_train = config['ev_session_months_train']
    ev_session_months_test = config['ev_session_months_test']
    ev_utility_coef_mean = config['ev_utility_coef_mean']
    ev_utility_coef_scale = config['ev_utility_coef_scale']
    apply_gaussian_noise = config['apply_gaussian_noise']

    # Pecan Street data
    with open(path_to_data + '/pecanstreet/metadata_dict.pickle', 'rb') as f:
        ps_metadata = pickle.load(f, )

    ps_all_dates = None
    for hid in ps_metadata:
        hid_dates = set(ps_metadata[hid]['dates'])
        ps_all_dates = hid_dates if ps_all_dates is None else hid_dates & ps_all_dates

    ps_dates_train, ps_dates_test = split_dates_train_and_test_monthly(ps_all_dates, days_per_month_train)
    ps_device_type_to_column = {'pv': 'solar', 'load': 'usage'}

    # Canopy data
    path_to_canopy_data = path_to_data + '/pvdata.nist.gov/'
    with open(path_to_data + '/pvdata.nist.gov/metadata_dict.pickle', 'rb') as f:
        canopy_metadata = pickle.load(f, )
    canopy_all_dates = canopy_metadata['dates']
    canopy_dates_train, canopy_dates_test = split_dates_train_and_test_monthly(canopy_all_dates, days_per_month_train)
    canopy_device_type_to_column = {'pv': 'InvPDC_kW_Avg', }
    canopy_solar_rated_power = canopy_metadata['solar_rated_power']

    # ElaadNL data
    path_to_elaadnl_data = path_to_data + '/elaadnl/'
    with open(path_to_data + '/elaadnl/metadata_dict.pickle', 'rb') as f:
        elaadnl_metadata = pickle.load(f, )
    ev_arrivals_per_minute = elaadnl_metadata['arrival_counts']

    # New York price data
    path_to_price_data = path_to_data + '//newyork_price/'
    with open(path_to_price_data + '/metadata_dict.pickle', 'rb') as f:
        price_metadata = pickle.load(f, )

    price_all_dates = price_metadata['dates']
    price_dates_train, price_dates_test = split_dates_train_and_test_monthly(price_all_dates, days_per_month_train)
    price_solar_rated_power = None
    price_device_type_to_column = {'feeder': 'price', }

    # Creating samplers
    pv_samplers_dict = {hid: TimeSeriesSampler(t0_hr, dt_min, path_to_data + 'pecanstreet/houses/' + hid + '.csv',
                                               ps_metadata[hid]['solar_rated_power'],
                                               ps_device_type_to_column, apply_gaussian_noise=apply_gaussian_noise)
                        for hid in ps_metadata}

    canopy_sampler = TimeSeriesSampler(t0_hr, dt_min, path_to_data + '/pvdata.nist.gov/processed_data.csv',
                                       canopy_solar_rated_power, canopy_device_type_to_column,
                                       apply_gaussian_noise=apply_gaussian_noise)

    price_sampler = TimeSeriesSampler(t0_hr, dt_min, path_to_data + '/newyork_price/price.csv',
                                      price_solar_rated_power, price_device_type_to_column,
                                      apply_gaussian_noise=apply_gaussian_noise)

    ev_sampler = EVSessionSampler(t0_hr, dt_min, ev_dt_min, ev_sampling_dt_min,
                                  path_to_data + '/elaadnl/charging_sessions.csv',
                                  ev_utility_coef_mean, ev_utility_coef_scale,
                                  apply_gaussian_noise=apply_gaussian_noise)

    return (pv_samplers_dict, ps_metadata, canopy_sampler, canopy_metadata,
            price_sampler, price_metadata, ev_sampler, elaadnl_metadata)
