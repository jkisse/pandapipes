import pandapipes as ps

import pandas as pd
from os.path import join
from pandapipes.plotting import simple_plot


def create_example_gas_grid():
    fluid = 'hgas'  # water, lgas, hydrogen
    net = ps.create_empty_network('pandapipes workshop - net 1', fluid=fluid)

    input_dir = r'net_data_gas'
    in_junctions = pd.read_csv(join(input_dir, 'example_net-junctions.CSV'))
    in_pipes = pd.read_csv(join(input_dir, 'example_net-pipes.CSV'))
    in_sinks = pd.read_csv(join(input_dir, 'example_net-sinks.CSV'))



    geodata = in_junctions[['lon', 'lat']].values

    ps.create_junctions(net, nr_junctions=31, pn_bar=1, tfluid_k=283.15,
                             height_m=in_junctions['height'], geodata=geodata)

    ps.create_pipes_from_parameters(net, in_pipes['from_junction'], in_pipes['to_junction'],
                                         length_km=in_pipes['length_km'], diameter_m=0.05, k_mm=0.2)

    # alternatively, with standard types from 
    # https://pandapipes.readthedocs.io/en/develop/standard_types/std_types_in_pandapipes.html
    # ps.create_pipes(net, in_pipes['from_junction'], in_pipes['to_junction'],
    #                 std_type='50_PE_100_SDR_11',  length_km=in_pipes['length_km'])

    ps.create_sinks(net, in_sinks['junction'], mdot_kg_per_s=0.01) #in_sinks['m_dot'])

    ps.create_ext_grid(net, junction=0, p_bar=1, t_k=283.15)
    ps.create_ext_grid(net, junction=14, p_bar=1, t_k=283.15)

    ps.create_valves(net, [4, 9, 22], [16, 18, 28], 0.05)

    ps.pipeflow(net)
    return net


if __name__ == '__main__':
    #%%
    gnet = create_example_gas_grid()

    simple_plot(gnet, plot_sinks=True, plot_sources=True)

    ps.to_json(gnet, "workshop_example_gas.json")

    print('pressure:')
    print(gnet.res_junction.p_bar)
    print('velocity:')
    print(gnet.res_pipe.v_mean_m_per_s)

#%% change valve status
    gnet.valve.opened = False
    ps.pipeflow(gnet)

    print('pressure:')
    print(gnet.res_junction.p_bar)
    print('velocity:')
    print(gnet.res_pipe.v_mean_m_per_s)

#%% change gas to hydrogen

    hhv_h2 = 38.4 # kWh/kg
    hhv_hgas = 14.6 # kWh/kg

    # gnet.sink.scaling = hhv_hgas/hhv_h2
    gnet.sink.mdot_kg_per_s = gnet.sink.mdot_kg_per_s * hhv_hgas/hhv_h2

    ps.create_fluid_from_lib(gnet, 'hydrogen', True)
    ps.pipeflow(gnet)

    print('pressure:')
    print(gnet.res_junction.p_bar)
    print('velocity:')
    print(gnet.res_pipe.v_mean_m_per_s)
