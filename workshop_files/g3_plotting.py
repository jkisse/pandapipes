import seaborn
import numpy as np
import pandapipes as ps

import matplotlib.pyplot as plt
import pandapipes.plotting as plot
from pandapipes.plotting import simple_plot
from pandapower.plotting import cmap_continuous, create_annotation_collection

from g2_create_minimal_example import create_example_gas_grid


def create_pandapipes_collections(net, show_junction_ID=False, label_size=1):
    colors = seaborn.color_palette('colorblind')
    jc = plot.create_junction_collection(net, color=colors[0])
    pc = plot.create_pipe_collection(net, color=colors[1])
    if hasattr(net, 'valve'):
        vc = plot.create_valve_collection(net, color=colors[2])
    else:
        vc = None
    sc = plot.create_sink_collection(net, color=colors[3])
    ec = plot.create_ext_grid_collection(net, color=colors[4])
    # tuples of all junction coords
    if show_junction_ID:
        coords = net.junction_geodata[['x', 'y']].values
        jic = create_annotation_collection(size=label_size,
                                           texts=np.char.mod('%.0f', net.junction.index),
                                           coords=coords, zorder=150, color='k')
    else:
        jic = None
    collections = [jc, pc, vc, sc, ec, jic]
    return collections


def plot_gas_results(net, junction_size=15):

    sic = plot.create_sink_collection(net, patch_edgecolor='grey', line_color='grey')
    if hasattr(net, 'source'):
        src = plot.create_source_collection(net, patch_edgecolor='grey',
                                        line_color='grey')
    else:
        src = None
    ec = plot.create_ext_grid_collection(net)
    if hasattr(net, 'valve'):
        vc = plot.create_valve_collection(net, color='grey')
    else:
        vc = None


    # color map for pressure
    max_p = net.res_junction.p_bar.max()
    cmap_list_p=[(0, "red"), (max_p/2, "yellow"), (max_p, "green")]
    cmap_p, norm_p = cmap_continuous(cmap_list_p)

    jc = plot.create_junction_collection(net, size=junction_size, cmap=cmap_p, norm=norm_p,
                                         z=net.res_junction.p_bar,
                                         cbar_title="junction pressure [bar]")

    # color map for velocity
    cmap_list_p=[(0, "green"), (6, "yellow"), (12, "red")]
    cmap_v, norm_v = cmap_continuous(cmap_list_p)
    pc = plot.create_pipe_collection(net, linewidths=1,
                                     cmap=cmap_v, norm=norm_v,
                                     z=net.res_pipe.v_mean_m_per_s.abs(),
                                     cbar_title="mean gas velocity [m/s]")

    plot.draw_collections([sic, src, ec, jc, pc, vc], figsize=(8,6))



if __name__ == '__main__':
    # import pandapower
    # enet = minimal_example_power_grid()
    # pandapower.to_json(enet, 'workshop_example_power.json')



    net = create_example_gas_grid()
    ps.to_json(net, 'workshop_example_gas.json')

    simple_plot(net, plot_sinks=True)

    collections = create_pandapipes_collections(net)

    plot.draw_collections(collections)

    plt.show()


    ps.pipeflow(net)
    plot_gas_results(net)