import numpy as np
import pandapipes as ps
import pandapower as pp
import pandas as pd
from matplotlib import pyplot as plt
from pandapower.control import ConstControl
from pandapower.plotting import draw_collections

from e3_plotting import create_pandapower_collections
from g2_create_minimal_example import create_example_gas_grid
from e2_minimal_example import minimal_example_power_grid

from pandapipes.multinet.control.controller.multinet_control import coupled_p2g_const_control, \
    coupled_g2p_const_control, P2GControlMultiEnergy, G2PControlMultiEnergy
from pandapipes.multinet.timeseries.run_time_series_multinet import run_timeseries
from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapipes.multinet.control.run_control_multinet import run_control

from g3_plotting import create_pandapipes_collections


# enet = minimal_example_power_grid()
# gnet = create_example_gas_grid()


enet = pp.from_json('workshop_example_power.json')
gnet = ps.from_json('workshop_example_gas.json')

mn = create_empty_multinet('coupled_networks')
add_net_to_multinet(mn, enet, 'power')
add_net_to_multinet(mn, gnet, 'gas')


p2g_bus = 30
p2g_junction = 30

p2g_load = pp.create_load(enet, bus=p2g_bus, p_mw=0.05, name='P2G unit')
p2g_source = ps.create_source(gnet, junction=p2g_junction, mdot_kg_per_s=0, name='P2G unit')

P2GControlMultiEnergy(mn, p2g_load, p2g_source, efficiency=0.7,
                      name_power_net='power', name_gas_net='gas')

pp.runpp(enet)
ps.pipeflow(gnet)
run_control(mn)

#%% combined plots:
gnet.junction_geodata.y -= 4
ps_coll = create_pandapipes_collections(gnet)
pp_coll = create_pandapower_collections(enet, 5)
draw_collections(ps_coll + pp_coll)

plt.show()

#%% time series

def prepare_data_power():
    data_pv = pd.read_excel(r'net_data_power/time_series.xlsx', sheet_name='pv_solar',
                            index_col='time')
    data_ho = pd.read_excel(r'net_data_power/time_series.xlsx', sheet_name='households', index_col='time')
    data_ho = data_ho.loc[data_pv.index]
    return data_pv.loc[:, data_pv.columns.str.contains('nom')], data_ho

def prepare_data_gas(n_timesteps=10):
    profiles = pd.DataFrame()
    profiles['wind farm'] = np.random.random(n_timesteps) * 300
    profiles['power to gas consumption'] = np.random.random(n_timesteps) * 200
    profiles['gas to power consumption'] = [3.6187] * n_timesteps
    ds = DFData(profiles)
    return profiles, ds

    profiles, ds = create_data_source(10)
    ows = create_output_writers(mn, 10)

def couple_data_network(net):
    data_pv, data_ho = prepare_data_power()
    profs_ho = np.random.choice(['A', 'B', 'C'], len(net.load.index))
    profs_pv = np.random.choice(['pv_south_nom', 'pv_southwest_nom'], len(net.sgen.index))
    profs_ho_p = ['H0-%s_pload' %x for x in profs_ho]
    profs_ho_q = ['H0-%s_qload' %x for x in profs_ho]
    data_source_p = data_ho.loc[:, data_ho.columns.str.contains('pload')] * 0.06
    data_source_q = data_ho.loc[:, data_ho.columns.str.contains('qload')] * 0.02
    data_source_pv = data_pv * 0.03

    ConstControl(net, 'load', 'p_mw', net.load.index, profs_ho_p, DFData(data_source_p))
    ConstControl(net, 'load', 'q_mvar', net.load.index, profs_ho_q, DFData(data_source_q))
    ConstControl(net, 'sgen', 'p_mw', net.sgen.index, profs_pv, DFData(data_source_pv))

    ow = OutputWriter(net, data_pv.index, r'results', output_file_type='.csv')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_trafo', 'loading_percent')

    run_timeseries(net, data_pv.index)
    # net.output_writer.loc[0, 'object'].output['res_bus.vm_pu']
    return net