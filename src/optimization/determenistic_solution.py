from src.optimization.util import dict_to_matrix
from collections import defaultdict
from pyomo.environ import *


def compute_deterministic_solution(dt_min, evs_dict, u, p_lbs, p_ubs, v_lbs, v_ubs, conductance_matrix, i_max_matrix,
                                   lossless=False, tee=False):
    p_lbs, p_ubs = p_lbs * 1000, p_ubs * 1000  # Transform powers to W from  kW
    for ev_ind in evs_dict:
        (d_ind, t_arr, t_dep, demand, u_i) = evs_dict[ev_ind]
        evs_dict[ev_ind] = (d_ind, t_arr, t_dep, demand * 1000, u_i)
    # Compute dictionary that maps device index to ev index
    device_to_evs_dict = defaultdict(lambda: dict())
    for ev_ind in evs_dict:
        (d_ind, t_arr, t_dep, demand, u_i) = evs_dict[ev_ind]
        for ti in range(t_arr, t_dep):
            device_to_evs_dict[d_ind][ti] = ev_ind
    model = ConcreteModel()
    model.devices = Set(initialize=range(u.shape[1]))
    model.timesteps = Set(initialize=range(u.shape[0]))
    model.p = Var(model.timesteps, model.devices)
    model.v = Var(model.timesteps, model.devices)
    # Lower and upper bounds
    for i in model.devices:
        for t in model.timesteps:
            model.p[t, i].setlb(p_lbs[t, i])
            model.p[t, i].setub(p_ubs[t, i])
            model.v[t, i].setlb(v_lbs[t, i])
            model.v[t, i].setub(v_ubs[t, i])
    # Power balance
    model.power_balance = ConstraintList()
    for i in model.devices:
        for t in model.timesteps:
            v_i = model.v[t, i] if not lossless else v_ubs[t, 0]
            p_i = -v_i * sum([conductance_matrix[i, j] * (model.v[t, i] - model.v[t, j])
                              for j in model.devices if i != j])
            model.power_balance.add(model.p[t, i] == p_i)
    # Line currents
    model.line_constraints = ConstraintList()
    for i in model.devices:
        for j in model.devices:
            for t in model.timesteps:
                if conductance_matrix[i, j] > 0:
                    i_line = conductance_matrix[i, j] * (model.v[t, i] - model.v[t, j])
                    i_line_max = i_max_matrix[i, j]
                    model.line_constraints.add(inequality(-i_line_max, i_line, i_line_max))
    # SOC constraints
    model.soc_constraints = ConstraintList()
    for ev_ind, (d_ind, t_arr, t_dep, demand, u_i) in evs_dict.items():
        for t_before_dep in range(t_arr, t_dep):
            soc_next_step = sum([model.p[ti, d_ind] for ti in range(t_arr, t_before_dep + 1)]) * dt_min / 60
            model.soc_constraints.add(inequality(0, soc_next_step, demand))
    # Objective
    per_device_utility = []
    for i in model.devices:
        for t in model.timesteps:
            val = u[t, i] * model.p[t, i]
            per_device_utility.append(val)
    model.f = Objective(sense=maximize, expr=sum(per_device_utility))
    if lossless:
        solver = SolverFactory('glpk')
    else:
        solver = SolverFactory('ipopt')
    solver.solve(model, tee=tee)
    p = dict_to_matrix(model.p, model.timesteps.data(), model.devices.data()) / 1000
    v = dict_to_matrix(model.v, model.timesteps.data(), model.devices.data())
    return p, v, model
