# IDM Project Electric Vehicle Charging Problem

This repository contains code for simulating the EV charging problem in DC grids.

## Installation

```
pip install -r requirements.txt
```

On macOS also run: `brew install ipopt`.

## Some documentation

### 1. `src/devices/`
Different devices that can be modeled in the EVCP problem are implemented using the Device class.
Current implementation includes the following devices:
1. `devices/feeder/FeederDevice` - a connection to the external power grid. Its generation capacity p_min is constant, and power price `utility_coef` is time-dependent and stochastic.
2. `devices/pv/PVDevice` - a solar panel connected to the grid. Provides free power (`utility_coef=0`), but its generation capacity `p_min` is stochastic.
3. `devices/load/LoadDevice` - an inflexible load (e.g., a household) in the grid. Its demand is stochastic and must be fulfilled (`p_min=p_max>0`) at each time step.
4. `devices/ev_charger/EVChargerDevice` - an EV charging station where EVs arrive stochastically.
5. `devices/ev/EV` - an EV entity. EV is implemented without using Devices class, as it is considered to be an exogenous element. Each EV has arrival and departure times (`t_arr`, `t_dep`), demand (`soc_goal`), and utility coefficient (`utility_coef`). Charging EV with 1 kWh of energy increases social welfare by 1.

Importantly, all devices use power (and hence `utility_coef`) in kW and voltage in V. 

### 2.`src/samplers/`
Samplers are interfaces between the devices and corresponding datasets. Each device has an assigned sampler which it uses to sample values of the uncertainties.
1. `samplers/time_series_sampler/TimeSeriesSampler` is used to sample data stored in the time series format. It can deal with power price dataset `/data/newyork_price`, demand and PV generation data from `/data/pecanstreet` and PV generation data from `/data/pvdata.nist.gov`/.
2. `samplers/ev_session_sampler/EVSessionSampler` is used to sample EV sessions. It uses the `/data/elaadnl/` and externally provided parameter arrival_rate to dataset to sample arrival and departure times of the EVs. `utility_coef` is sampled using normal distribution.

### 3.`src/environments/`
Environment combines list of devices with their topological properties (conductance matrix, line constraints matrix) and is used to run the simulation.
