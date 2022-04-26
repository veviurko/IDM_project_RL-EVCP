import pandas as pd
import numpy as np


def hour_and_minute_to_t_str(hour, minute):
    return ('0%d:' % hour if hour < 10 else '%d:' % hour) + ('0%d' % minute if minute < 10 else '%d' % minute)


def t_hr_to_t_str(t_hr):
    hour = int(t_hr)
    minute = np.round(t_hr * 60) - hour * 60
    return hour_and_minute_to_t_str(hour, minute)


def t_min_to_t_str(t_min):
    hour = int(t_min // 60)
    minute = t_min - np.round(hour * 60)
    return hour_and_minute_to_t_str(hour, minute)


def t_min_to_t_hr(t_min):
    return t_min / 60


def t_hr_to_t_min(t_hr):
    return int(t_hr * 60)


def t_str_to_t_hr(t_str):
    return int(t_str[:2]) + int(t_str[3:]) / 60


def t_str_to_t_min(t_str):
    return int(t_str[:2]) * 60 + int(t_str[3:])


def create_timesteps_hr(t0_hr, dt_min, round_to=4):
    return list(
        ((np.arange(0, 24, dt_min / 60) + t0_hr) % 24).round(round_to)
    )


def round_t_hr(t_hr, dt_min):
    return dt_min * (t_hr * 60 // dt_min) / 60


def shift_date(datetime_string, t0_hr):
    datetime = pd.Timestamp(datetime_string)
    t_hr = datetime.hour + datetime.minute / 60
    if t_hr < t0_hr:
        date_new_str = str((datetime - pd.Timedelta(1, unit='D')))[:10]
    else:
        date_new_str = str(datetime)[:10]
    return date_new_str


def split_dates_train_and_test_monthly(all_dates, n_days_train):
    day_str_train = ['0%d' % d if d < 10 else '%d' % d for d in range(n_days_train)]
    dates_train = [date for date in all_dates if date[-2:] in day_str_train]
    dates_test = [date for date in all_dates if date[-2:] not in day_str_train]
    return dates_train, dates_test
