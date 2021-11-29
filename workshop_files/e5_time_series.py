import pandas as pd
from e2_minimal_example import minimal_example_power_grid
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries
import numpy as np
import pandapower as pp


def prepare_data():
    data_pv = pd.read_excel(r'net_data_power/time_series.xlsx', sheet_name='pv_solar',
                            index_col='time')
    data_ho = pd.read_excel(r'net_data_power/time_series.xlsx', sheet_name='households', index_col='time')
    data_ho = data_ho.loc[data_pv.index]
    return data_pv.loc[:, data_pv.columns.str.contains('nom')], data_ho

def couple_data_network():
    net = minimal_example_power_grid()
    data_pv, data_ho = prepare_data()
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

def compare_peak_time_series(net):
    #time series simulation
    res = net.output_writer.loc[0, 'object'].np_results
    print(res['res_bus.vm_pu'].max())
    print(res['res_bus.vm_pu'].min())
    print(res['res_line.loading_percent'].max())
    print(res['res_trafo.loading_percent'].max())

    #peak load case
    net = minimal_example_power_grid()
    net.load.scaling = 1.0
    net.sgen.scaling = 0.0
    pp.runpp(net)
    print(net.res_bus.vm_pu.max())
    print(net.res_bus.vm_pu.min())
    print(net.res_line.loading_percent.max())
    print(net.res_trafo.loading_percent.max())

    #peak feed-in case
    net = minimal_example_power_grid()
    net.load.scaling = 0.1
    net.sgen.scaling = 1.
    pp.runpp(net)
    print(net.res_bus.vm_pu.max())
    print(net.res_bus.vm_pu.min())
    print(net.res_line.loading_percent.max())
    print(net.res_trafo.loading_percent.max())


if __name__ == '__main__':
    net = couple_data_network()
    compare_peak_time_series(net)