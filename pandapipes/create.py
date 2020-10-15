# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from packaging import version
from pandapipes.component_models.auxiliaries.component_toolbox import add_new_component
from pandapipes.pandapipes_net import pandapipesNet, get_default_pandapipes_structure
from pandapipes.properties import call_lib
from pandapipes.properties.fluids import _add_fluid_to_net
from pandapower.auxiliary import get_free_id, _preserve_dtypes
from pandapipes.properties.fluids import Fluid
from pandapipes.std_types.std_type import PumpStdType, add_basic_std_types, add_pump_std_type, \
    load_std_type
from pandapipes.std_types.std_type_toolbox import regression_function
from pandapipes.component_models import Junction, Sink, Source, Pump, Pipe, ExtGrid, \
    HeatExchanger, Valve, CirculationPumpPressure, CirculationPumpMass

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_empty_network(name="", fluid=None, add_stdtypes=True):
    """
    This function initializes the pandapipes datastructure.

    :param name: Name for the network
    :type name: string, default None
    :param fluid: A fluid that can be added to the net from the start. Should be either of type\
            Fluid (c.f. pandapipes.properties.fluids.Fluid) or a string which refers to a standard\
            fluid type used to call `create_fluid_from_lib`. A fluid is required for pipeflow\
            calculations, but can also be added later.
    :type fluid: Fluid or str, default None
    :param add_stdtypes: Flag whether to add a dictionary of typical pump and pipe std types
    :type add_stdtypes: bool, default True
    :return: net - pandapipesNet with empty tables
    :rtype: pandapipesNet

    :Example:
        >>> net1 = create_empty_network("my_first_pandapipesNet", "lgas")
        >>> net2 = create_empty_network()

    """
    net = pandapipesNet(get_default_pandapipes_structure())
    add_new_component(net, Junction, True)
    add_new_component(net, Pipe, True)
    add_new_component(net, ExtGrid, True)
    net['controller'] = pd.DataFrame(np.zeros(0, dtype=net['controller']), index=[])
    net['name'] = name
    if add_stdtypes:
        add_basic_std_types(net)

    if fluid is not None:
        if isinstance(fluid, Fluid):
            net["fluid"] = fluid
        elif isinstance(fluid, str):
            create_fluid_from_lib(net, fluid)
        else:
            logger.warning("The fluid %s cannot be added to the net Only fluids of type Fluid or "
                           "strings can be used." % fluid)
    return net


def create_junction(net, pn_bar, tfluid_k, height_m=0, name=None, index=None, in_service=True,
                    type="junction", geodata=None, **kwargs):
    """
    Adds one junction in table net["junction"]. Junctions are the nodes of the network that
    all other elements connect to.

    :param net: The pandapipes network in which the element is created
    :type net: pandapipesNet
    :param pn_bar: The nominal pressure in [bar]. Used as an initial value for pressure calculation.
    :type pn_bar: float
    :param tfluid_k: The fluid temperature in [K]. Used as parameter for gas calculations and as\
            initial value for temperature calculations.
    :type tfluid_k: float
    :param height_m: Height of node above sea level in [m]
    :type height_m: float, default 0
    :param name: The name for this junction
    :type name: string, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: boolean, default True
    :param type: not used yet - Designed for type differentiation on pandas lookups (e.g. household\
            connection vs. crossing)
    :type type: string, default "junction"
    :param geodata: Coordinates used for plotting
    :type geodata: (x,y)-tuple, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["junction"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_junction(net, pn_bar=5, tfluid_k=320)
    """
    add_new_component(net, Junction)

    if index and index in net["junction"].index:
        raise UserWarning("A junction with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["junction"])

    # store dtypes
    dtypes = net.junction.dtypes
    cols = ["name", "pn_bar", "tfluid_k", "height_m", "in_service", "type"]
    vals = [name, pn_bar, tfluid_k, height_m, bool(in_service), type]

    all_values = {col: val for col, val in zip(cols, vals)}
    all_values.update(kwargs)
    for col, val in all_values.items():
        net.junction.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.junction, dtypes)

    if geodata is not None:
        if len(geodata) != 2:
            raise UserWarning("geodata must be given as (x, y) tupel")
        net["junction_geodata"].loc[index, ["x", "y"]] = geodata

    return index


def create_sink(net, junction, mdot_kg_per_s, scaling=1., name=None, index=None, in_service=True,
                type='sink', **kwargs):
    """
    Adds one sink in table net["sink"].

    :param net: The net for which this sink should be created
    :type net: pandapipesNet
    :param junction: The index of the junction to which the sink is connected
    :type junction: int
    :param mdot_kg_per_s: The required mass flow
    :type mdot_kg_per_s: float, default None
    :param scaling: An optional scaling factor to be set customly
    :type scaling: float, default 1
    :param name: A name tag for this sink
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param type: Type variable to classify the sink
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["sink"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> new_sink_id = create_sink(net, junction=2, mdot_kg_per_s=0.1)

    """
    add_new_component(net, Sink)

    if junction not in net["junction"].index.values:
        raise UserWarning("Cannot attach to junction %s, junction does not exist" % junction)

    if index is None:
        index = get_free_id(net["sink"])

    if index in net["sink"].index:
        raise UserWarning("A sink with the id %s already exists" % index)

    # store dtypes
    dtypes = net.sink.dtypes

    cols = ["name", "junction", "mdot_kg_per_s", "scaling", "in_service", "type"]
    vals = [name, junction, mdot_kg_per_s, scaling, bool(in_service), type]
    all_values = {col: val for col, val in zip(cols, vals)}
    all_values.update(kwargs)
    for col, val in all_values.items():
        net.sink.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.sink, dtypes)

    return index


def create_source(net, junction, mdot_kg_per_s, scaling=1., name=None, index=None, in_service=True,
                  type='source', **kwargs):
    """
    Adds one source in table net["source"].

    :param net: The net for which this source should be created
    :type net: pandapipesNet
    :param junction: The index of the junction to which the source is connected
    :type junction: int
    :param mdot_kg_per_s: The required mass flow
    :type mdot_kg_per_s: float, default None
    :param scaling: An optional scaling factor to be set customly
    :type scaling: float, default 1
    :param name: A name tag for this source
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param type: Type variable to classify the source
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["source"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_source(net,junction=2,mdot_kg_per_s=0.1)

    """
    add_new_component(net, Source)

    if junction not in net["junction"].index.values:
        raise UserWarning("Cannot attach to junction %s, junction does not exist" % junction)

    if index is None:
        index = get_free_id(net["source"])

    if index in net["source"].index:
        raise UserWarning("A source with the id %s already exists" % index)

    # store dtypes
    dtypes = net.source.dtypes

    cols = ["name", "junction", "mdot_kg_per_s", "scaling", "in_service", "type"]
    vals = [name, junction, mdot_kg_per_s, scaling, bool(in_service), type]
    all_values = {col: val for col, val in zip(cols, vals)}
    all_values.update(kwargs)
    for col, val in all_values.items():
        net.source.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.source, dtypes)

    return index


def create_ext_grid(net, junction, p_bar, t_k, name=None, in_service=True, index=None, type="pt"):
    """
    Creates an external grid and adds it to the table net["ext_grid"]. It transfers the junction
    that it is connected to into a node with fixed value for either pressure, temperature or both
    (depending on the type). Usually external grids represent connections to other grids feeding
    the given pandapipesNet.

    :param net: The net that the external grid should be connected to
    :type net: pandapipesNet
    :param junction: The junction to which the external grid is connected
    :type junction: int
    :param p_bar: The pressure of the external grid
    :type p_bar: float
    :param t_k: The fixed temperature at the external grid
    :type t_k: float, default 285.15
    :param name: A name tag for this ext_grid
    :type name: str, default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :param type: The external grid type denotes the values that are fixed at the respective node:\n
            - "p": The pressure is fixed, the node acts as a slack node for the mass flow.
            - "t": The temperature is fixed and will not be solved for, but is assumed as the node's mix temperature. Please note that pandapipes cannot check for inconsistencies in the formulation of heat transfer equations yet. \n
            - "pt": The external grid shows both "p" and "t" behavior.
    :type type: str, default "pt"
    :type index: int, default None
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_ext_grid(net, junction=2, p_bar=100, t_k=293.15)

    """
    add_new_component(net, ExtGrid)

    if not type in ["p", "t", "pt"]:
        logger.warning("no proper type was chosen.")

    if junction not in net["junction"].index.values:
        raise UserWarning("Cannot attach to junction %s, junction does not exist" % junction)

    if index is not None and index in net["ext_grid"].index:
        raise UserWarning("An external grid with with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["ext_grid"])

    # store dtypes
    dtypes = net.ext_grid.dtypes

    net.ext_grid.loc[index, ["name", "junction", "p_bar", "t_k", "in_service", "type"]] = \
        [name, junction, p_bar, t_k, bool(in_service), type]

    # and preserve dtypes
    _preserve_dtypes(net.ext_grid, dtypes)
    return index


def create_heat_exchanger(net, from_junction, to_junction, diameter_m, qext_w, loss_coefficient=0,
                          name=None, index=None, in_service=True, type="heat_exchanger", **kwargs):
    """
    Creates a heat exchanger element in net["heat_exchanger"] from heat exchanger parameters.

    :param net: The net within this heat exchanger should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the heat exchanger will be\
            connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the heat exchanger will be\
            connected with
    :type to_junction: int
    :param diameter_m: The heat exchanger inner diameter in [m]
    :type diameter_m: float
    :param qext_w: External heat feed-in through the heat exchanger in [W]
    :type qext_w: float, default 0.0
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float
    :param name: The name of the heat exchanger
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type: Not used yet
    :type type: str
    :param kwargs: Additional keyword arguments will be added as further columns to the\
                    net["heat_exchanger"] table
    :return: index - The unique ID of the created heat exchanger
    :rtype: int

    :Example:
        >>> create_heat_exchanger(net, from_junction=0, to_junction=1, diameter_m=40e-3, qext_w=2000)
    """
    add_new_component(net, HeatExchanger)

    # check if junction exist to attach the heat exchanger to
    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Heat exchanger %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["heat_exchanger"])

    if index in net["heat_exchanger"].index:
        raise UserWarning("A heat exchanger with index %s already exists" % index)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "diameter_m": diameter_m, "qext_w": qext_w, "loss_coefficient": loss_coefficient,
         "in_service": bool(in_service), "type": type}
    v.update(kwargs)

    # store dtypes
    dtypes = net.heat_exchanger.dtypes

    for col, val in v.items():
        net.heat_exchanger.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.heat_exchanger, dtypes)

    return index


def create_pipe(net, from_junction, to_junction, std_type, length_km, k_mm=1, loss_coefficient=0,
                sections=1, alpha_w_per_m2k=0., text_k=293, qext_w=0., name=None, index=None,
                geodata=None, in_service=True, type="pipe", **kwargs):
    """
    Creates a pipe element in net["pipe"] from pipe parameters.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pipe will be connected to
    :type from_junction: int
    :param to_junction: ID of the junction on the other side to which the pipe will be connected to
    :type to_junction: int
    :param std_type: Name of standard type
    :type std_type: str
    :param length_km: Length of the pipe in [km]
    :type length_km: float
    :param k_mm: Pipe roughness in [mm]
    :type k_mm: float, default 1
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param text_k: Ambient temperature of pipe in [K]
    :type text_k: float, default 293
    :param qext_w: External heat feed-in to the pipe in [W]
    :type qext_w: float, default 0
    :param name: A name tag for this pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param geodata: The coordinates of the pipe. The first row should be the coordinates of\
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe.
    :type geodata: array, shape=(,2L), default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of pipes (e.g. below or above ground)
    :type type: str, default "pipe"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_pipe(net,from_junction=0,to_junction=1,std_type='315_PE_80_SDR_17',length_km=1)

    """
    add_new_component(net, Pipe)

    # check if junction exist to attach the pipe to
    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pipe %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["pipe"])

    if index in net["pipe"].index:
        raise UserWarning("A pipe with index %s already exists" % index)

    _check_std_type(net, std_type, "pipe", "create_pipe")
    pipe_parameter = load_std_type(net, std_type, "pipe")
    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": std_type, "length_km": length_km, "diameter_m":
             pipe_parameter["inner_diameter_mm"] / 1000, "k_mm": k_mm,
         "loss_coefficient": loss_coefficient, "alpha_w_per_m2k": alpha_w_per_m2k,
         "sections": sections, "in_service": bool(in_service), "type": type, "qext_w": qext_w,
         "text_k": text_k}
    v.update(kwargs)

    # store dtypes
    dtypes = net.pipe.dtypes

    for col, val in v.items():
        net.pipe.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.pipe, dtypes)

    if geodata is not None:
        net["pipe_geodata"].at[index, "coords"] = geodata

    return index


def create_pipe_from_parameters(net, from_junction, to_junction, length_km, diameter_m, k_mm=1,
                                loss_coefficient=0, sections=1, alpha_w_per_m2k=0., text_k=293,
                                qext_w=0., name=None, index=None, geodata=None, in_service=True,
                                type="pipe", **kwargs):
    """
    Creates a pipe element in net["pipe"] from pipe parameters.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pipe will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side to which the pipe will be connected to
    :type to_junction: int
    :param length_km: Length of the pipe in [km]
    :type length_km: float
    :param diameter_m: The pipe diameter in [m]
    :type diameter_m: float
    :param k_mm: Pipe roughness in [mm]
    :type k_mm: float, default 1
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param qext_w: external heat feed-in to the pipe in [W]
    :type qext_w: float, default 0
    :param text_k: Ambient temperature of pipe in [K]
    :type text_k: float, default 293
    :param name: A name tag for this pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param geodata: The coordinates of the pipe. The first row should be the coordinates of\
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe
    :type geodata: array, shape= (,2L), default None
    :param in_service: True for in service, false for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of pipes (e.g. below or above ground)
    :type type: str, default "pipe"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_pipe_from_parameters(net,from_junction=0,to_junction=1,length_km=1,diameter_m=40e-3)

    """
    add_new_component(net, Pipe)

    # check if junction exist to attach the pipe to
    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pipe %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["pipe"])

    if index in net["pipe"].index:
        raise UserWarning("A pipe with index %s already exists" % index)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": None, "length_km": length_km, "diameter_m": diameter_m, "k_mm": k_mm,
         "loss_coefficient": loss_coefficient, "alpha_w_per_m2k": alpha_w_per_m2k,
         "sections": sections, "in_service": bool(in_service),
         "type": type, "qext_w": qext_w, "text_k": text_k}
    if 'std_type' in kwargs:
        raise UserWarning('you have defined a std_type, however, using this function you can only'
                          'create a pipe setting specific, individual parameters. If you want to '
                          'create a pipe from net.std_type please use create_pipe')
    v.update(kwargs)

    # store dtypes
    dtypes = net.pipe.dtypes

    for col, val in v.items():
        net.pipe.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.pipe, dtypes)

    if geodata is not None:
        net["pipe_geodata"].at[index, "coords"] = geodata

    return index


def create_valve(net, from_junction, to_junction, diameter_m, opened=True, loss_coefficient=0,
                 name=None, index=None, type='valve', **kwargs):
    """
    Creates a valve element in net["valve"] from valve parameters.

    :param net: The net for which this valve should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the valve will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the valve will be connected with
    :type to_junction: int
    :param diameter_m: The valve diameter in [m]
    :type diameter_m: float
    :param opened: Flag to show if the valve is opened and allows for fluid flow or if it is closed\
            to block the fluid flow.
    :type opened: bool, default True
    :param loss_coefficient: The pressure loss coefficient introduced by the valve shape
    :type loss_coefficient: float, default 0
    :param name: A name tag for this valve
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param type: An identifier for special types of valves
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["valve"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_valve(net, 0, 1, diameter_m=4e-3, name="valve1")

    """
    add_new_component(net, Valve)

    # check if junction exist to attach the pipe to
    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Valve %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["valve"])

    if index in net["valve"].index:
        raise UserWarning("A valve with index %s already exists" % index)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "diameter_m": diameter_m,
         "opened": opened, "loss_coefficient": loss_coefficient, "type": type}
    v.update(kwargs)
    # store dtypes
    dtypes = net.valve.dtypes

    for col, val in v.items():
        net.valve.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.valve, dtypes)

    return index


def create_pump(net, from_junction, to_junction, std_type, name=None, index=None, in_service=True,
                type="pump", **kwargs):
    """
    Adds one pump in table net["pump"].

    :param net: The net within this pump should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pump will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pump will be connected with
    :type to_junction: int
    :param std_type: There are currently three different std_types. This std_types are P1, P2, P3.\
            Each of them describes a specific pump behaviour setting volume flow and pressure in\
            context.
    :type std_type: string, default None
    :param name: A name tag for this pump
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type:  Type variable to classify the pump
    :type type: str, default "pump"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pump"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    EXAMPLE:
        >>> create_pump(net, 0, 1, std_type="P1")

    """
    add_new_component(net, Pump)

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pump %s tries to attach to non-existing junction %s" % (name, b))

    if index is None:
        index = get_free_id(net["pump"])
    if index in net["pump"].index:
        raise UserWarning("A pump with the id %s already exists" % id)

    # store dtypes
    dtypes = net.pump.dtypes

    _check_std_type(net, std_type, "pump", "create_pump")
    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": std_type, "in_service": bool(in_service), "type": type}
    v.update(kwargs)
    # and preserve dtypes
    for col, val in v.items():
        net.pump.at[index, col] = val
    _preserve_dtypes(net.pump, dtypes)

    return index


def create_pump_from_parameters(net, from_junction, to_junction, new_std_type_name,
                                pressure_list=None, flowrate_list=None, reg_polynomial_degree=None,
                                poly_coefficents=None, name=None, index=None, in_service=True,
                                type="pump", **kwargs):
    """
    Adds one pump in table net["pump"].

    :param net: The net within this pump should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pump will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pump will be connected with
    :type to_junction: int
    :param new_std_type_name: Set a name for your pump. You will find your definied pump under
            std_type in your net. The name will be given under std_type in net.pump.
    :type new_std_type_name: string
    :param pressure_list: This list contains measured pressure supporting points required\
            to define and determine the dependencies of the pump between pressure and volume flow.\
            The pressure must be given in [bar]. Needs to be defined only if no pump of standard\
            type is selected.
    :type pressure_list: list, default None
    :param flowrate_list: This list contains the corresponding flowrate values to the given\
            pressure values. Thus the length must be equal to the pressure list. Needs to be\
            defined only if no pump of standard type is selected. ATTENTION: The flowrate values\
            are given in :math:`[\\frac{m^3}{h}]`.
    :type flowrate_list: list, default None
    :param reg_polynomial_degree: The degree of the polynomial fit must be defined if pressure\
            and flowrate list are given. The fit describes the behaviour of the pump (delta P /\
            volumen flow curve).
    :type reg_polynomial_degree: int, default None
    :param poly_coefficents: Alternatviely to taking measurement values and degree of polynomial
            fit, previously calculated regression parameters can also be given directly. It
            describes the dependency between pressure and flowrate.\
            ATTENTION: The determined parameteres must be retrieved by setting flowrate given\
            in :math:`[\\frac{m^3}{h}]` and pressure given in bar in context. The first entry in\
            the list (c[0]) is for the polynom of highest degree (c[0]*x**n), the last one for
            c*x**0.
    :type poly_coefficents: list, default None
    :param name: A name tag for this pump
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type:  type variable to classify the pump
    :type type: str, default "pump"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pump"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    EXAMPLE:
        >>> create_pump_from_parameters(net, 0, 1, 'pump1', pressure_list=[0,1,2,3],\
                                        flowrate_list=[0,1,2,3], reg_polynomial_degree=1)
        >>> create_pump_from_parameters(net, 0, 1, 'pump2', poly_coefficents=[1,0])

    """
    add_new_component(net, Pump)

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pump %s tries to attach to non-existing junction %s" % (name, b))

    if index is None:
        index = get_free_id(net["pump"])
    if index in net["pump"].index:
        raise UserWarning("A pump with the id %s already exists" % id)

    # store dtypes
    dtypes = net.pump.dtypes

    if pressure_list is not None and flowrate_list is not None and reg_polynomial_degree is not None:
        reg_par = regression_function(pressure_list, flowrate_list, reg_polynomial_degree)
        pump = PumpStdType(new_std_type_name, reg_par)
        add_pump_std_type(net, new_std_type_name, pump)
    elif poly_coefficents is not None:
        pump = PumpStdType(new_std_type_name, poly_coefficents)
        add_pump_std_type(net, new_std_type_name, pump)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": new_std_type_name, "in_service": bool(in_service), "type": type}
    v.update(kwargs)
    # and preserve dtypes
    for col, val in v.items():
        net.pump.at[index, col] = val
    _preserve_dtypes(net.pump, dtypes)

    return index


def create_circ_pump_const_pressure(net, from_junction, to_junction, p_bar, plift_bar,
                                    t_k=None, name=None, index=None, in_service=True, type="pt",
                                    **kwargs):
    """
    Adds one circulation pump with a constant pressure lift in table net["circ_pump_pressure"].

    :param net: The net within this pump should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pump will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pump will be connected with
    :type to_junction: int
    :param p_bar: Pressure set point
    :type p_bar: float
    :param plift_bar: Pressure lift induced by the pump
    :type plift_bar: float
    :param t_k: Temperature set point
    :type t_k: float
    :param name: Name of the pump
    :type name: str
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type: The pump type denotes the values that are fixed:\n
            - "p": The pressure is fixed.
            - "t": The temperature is fixed and will not be solved. Please note that pandapipes\
             cannot check for inconsistencies in the formulation of heat transfer equations yet.
            - "pt": The pump shows both "p" and "t" behavior.
    :type type: str, default "pt"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["circ_pump_pressure"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_circ_pump_const_pressure(net, 0, 1, p_bar=5, plift_bar=2, t_k=350, type="p")

    """

    add_new_component(net, CirculationPumpPressure)

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning(
                    "CirculationPumpPressure %s tries to attach to non-existing junction %s"
                    % (name, b))

    if index is None:
        index = get_free_id(net["circ_pump_pressure"])
    if index in net["circ_pump_pressure"].index:
        raise UserWarning("A CirculationPumpPressure with the id %s already exists" % id)

    # store dtypes
    dtypes = net.circ_pump_pressure.dtypes

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "p_bar": p_bar, "t_k": t_k, "plift_bar": plift_bar,
         "in_service": bool(in_service), "type": type}
    v.update(kwargs)
    # and preserve dtypes
    for col, val in v.items():
        net.circ_pump_pressure.at[index, col] = val
    _preserve_dtypes(net.circ_pump_pressure, dtypes)

    return index


def create_circ_pump_const_mass_flow(net, from_junction, to_junction, p_bar, mdot_kg_per_s,
                                     t_k=None, name=None, index=None, in_service=True,
                                     type="pt", **kwargs):
    """
    Adds one circulation pump with a constant mass flow in table net["circ_pump_mass"].

    :param net: The net within this pump should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pump will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pump will be connected with
    :type to_junction: int
    :param p_bar: Pressure set point
    :type p_bar: float
    :param mdot_kg_per_s: Constant mass flow, which is transported through the pump
    :type mdot_kg_per_s: float
    :param t_k: Temperature set point
    :type t_k: float
    :param name: Name of the pump
    :type name: str
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type: The pump type denotes the values that are fixed:\n
            - "p": The pressure is fixed.
            - "t": The temperature is fixed and will not be solved. Please note that pandapipes\
             cannot check for inconsistencies in the formulation of heat transfer equations yet.
            - "pt": The pump shows both "p" and "t" behavior.
    :type type: str, default "pt"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["circ_pump_mass"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_circ_pump_const_mass_flow(net, 0, 1, p_bar=5, mdot_kg_per_s=2, t_k=350, type="p")

    """

    add_new_component(net, CirculationPumpMass)

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("CirculationPumpMass %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["circ_pump_mass"])
    if index in net["circ_pump_mass"].index:
        raise UserWarning("A CirculationPumpMass with the id %s already exists" % id)

    # store dtypes
    dtypes = net.circ_pump_mass.dtypes

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "p_bar": p_bar, "t_k": t_k, "mdot_kg_per_s": mdot_kg_per_s,
         "in_service": bool(in_service), "type": type}
    v.update(kwargs)
    # and preserve dtypes
    for col, val in v.items():
        net.circ_pump_mass.at[index, col] = val
    _preserve_dtypes(net.circ_pump_mass, dtypes)

    return index


def create_junctions(net, nr_junctions, pn_bar, tfluid_k, heights_m=0, names=None, index=None,
                     in_service=True, types="junction", geodata=None, **kwargs):
    """
    Adds several junctions in table net["junction"] at once. Junctions are the nodes of the network
    that all other elements connect to.

    :param net: The pandapipes network in which the element is created
    :type net: pandapipesNet
    :param nr_junctions: Number of junctions to be created.
    :type nr_junctions: int
    :param pn_bar: The nominal pressure in [bar]. Used as an initial value for pressure calculation.
    :type pn_bar: Iterable or float
    :param tfluid_k: The fluid temperature in [K]. Used as parameter for gas calculations and as\
            initial value for temperature calculations.
    :type tfluid_k: Iterable or float
    :param heights_m: Heights of nodes above sea level in [m]
    :type heights_m: Iterable or float, default 0
    :param names: The names for these junctions
    :type names: Iterable or string, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param in_service: True for in_service or False for out of service
    :type in_service: Iterable or boolean, default True
    :param types: not used yet - Designed for type differentiation on pandas lookups (e.g. \
            household connection vs. crossing)
    :type types: Iterable or string, default "junction"
    :param geodata: Coordinates used for plotting
    :type geodata: Iterable of (x,y)-tuples, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["junction"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> create_junctions(net, 200, pn_bar=5, tfluid_k=320, height_m=np.arange(200))
    """
    add_new_component(net, Junction)

    index = _get_multiple_index_with_check(net, "junction", index, nr_junctions)
    entries = {"pn_bar": pn_bar,  "type": types, "tfluid_k": tfluid_k, "height_m": heights_m,
               "in_service": in_service, "name": names}
    entries.update(kwargs)
    _add_entries_to_table(net, "junction", index, entries)

    if geodata is not None:
        # works with a 2-tuple or a matching array
        net.junction_geodata = net.junction_geodata.append(pd.DataFrame(
            np.zeros((len(index), len(net.junction_geodata.columns)), dtype=int), index=index,
            columns=net.junction_geodata.columns))
        net.junction_geodata.loc[index, :] = np.nan
        net.junction_geodata.loc[index, ["x", "y"]] = geodata

    return index


def create_sinks(net, junctions, mdot_kg_per_s, scaling=1., names=None, index=None, in_service=True,
                 types='sink', **kwargs):
    """
    Adds several sinks in table net["sink"]. Arguments can be passed as one for all sinks or as \
    list containing values for each created sink.

    :param net: The net for which this sink should be created
    :type net: pandapipesNet
    :param junctions: The index of the junctions to which the sinks are connected
    :type junctions: Iterable(int)
    :param mdot_kg_per_s: The required mass flow
    :type mdot_kg_per_s: Iterable or float, default None
    :param scaling: An optional scaling factor to be set customly
    :type scaling: Iterable or float, default 1
    :param names: Name tags for the sinks
    :type names: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable or bool, default True
    :param types: Type variables to classify the sinks
    :type types: Iterable or str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["sink"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> new_sink_ids = create_sinks(net, junctions=[1, 5, 10], mdot_kg_per_s=[0.1, 0.05, 0.2])
    """
    add_new_component(net, Sink)

    _check_node_elements(net, junctions)
    index = _get_multiple_index_with_check(net, "sink", index, len(junctions))

    entries = {"junction": junctions, "mdot_kg_per_s": mdot_kg_per_s, "scaling": scaling,
               "in_service": in_service, "name": names, "type": types}
    entries.update(kwargs)
    _add_entries_to_table(net, "sink", index, entries)

    return index


def create_sources(net, junctions, mdot_kg_per_s, scaling=1., names=None, index=None,
                   in_service=True, types='source', **kwargs):
    """
    Adds several sources in table net["source"]. Arguments can be passed as one for all sources or \
    as list containing values for each created source.

    :param net: The net for which this source should be created
    :type net: pandapipesNet
    :param junctions: The index of the junctions to which the sources are connected
    :type junctions: Iterabl(int)
    :param mdot_kg_per_s: The required mass flow
    :type mdot_kg_per_s: Iterable or float, default None
    :param scaling: An optional scaling factor to be set customly
    :type scaling: Iterable or float, default 1
    :param names: Name tags for the sources
    :type names: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable or bool, default True
    :param types: Type variable to classify the sources
    :type types: Iterable or str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["source"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> new_source_ids = create_sources(net, junctions=[1, 5, 10], mdot_kg_per_s=[0.1, 0.05, 0.2])
    """
    add_new_component(net, Source)

    _check_node_elements(net, junctions)
    index = _get_multiple_index_with_check(net, "source", index, len(junctions))

    entries = {"junction": junctions, "mdot_kg_per_s": mdot_kg_per_s, "scaling": scaling,
               "in_service": in_service, "name": names, "type": types}
    entries.update(kwargs)
    _add_entries_to_table(net, "source", index, entries)

    return index


def create_pipes(net, from_junctions, to_junctions, std_type, lengths_km, k_mm=1,
                 loss_coefficients=0, sections=1, alpha_w_per_m2k=0., text_k=293, qext_w=0.,
                 names=None, index=None, geodata=None, in_service=True, types="pipe", **kwargs):
    """
    Convenience function for creating many pipes at once. Parameters 'from_junctions' and \
    'to_junctions' must be arrays of equal length. Other parameters may be either arrays of the \
    same length or single or values. In any case the line parameters are defined through a single \
    standard type, so all pipes have the same standard type.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junctions: IDs of the junctions on one side which the pipes will be connected to
    :type from_junctions: Iterable(int)
    :param to_junctions: IDs of the junctions on the other side to which the pipes will be \
            connected to
    :type to_junctions: Iterable(int)
    :param std_type: Name of standard type
    :type std_type: str
    :param lengths_km: Lengths of the pipes in [km]
    :type lengths_km: Iterable or float
    :param k_mm: Pipe roughness in [mm]
    :type k_mm: Iterable or float, default 1
    :param loss_coefficients: Additional pressure loss coefficients, introduced by e.g. bends
    :type loss_coefficients: Iterable or float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: Iterable or int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficients in [W/(m^2*K)]
    :type alpha_w_per_m2k: Iterable or float, default 0
    :param text_k: Ambient temperatures of pipes in [K]
    :type text_k: Iterable or float, default 293
    :param qext_w: External heat feed-in to the pipes in [W]
    :type qext_w: Iterable or float, default 0
    :param names: Name tags for these pipes
    :type names: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param geodata: The coordinates of the pipes. The first row should be the coordinates of\
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe.
    :type geodata: array, shape=(no_pipes,2L) or (,2L), default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable or bool, default True
    :param types: Identifiers for special types of pipes (e.g. below or above ground)
    :type types: Iterable or str, default "pipe"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> pipe_indices = create_pipes(net, from_junctions=[0, 2, 6], to_junctions=[1, 3, 7], \
                                        std_type='315_PE_80_SDR_17', lengths_km=[0.2, 1, 0.3])

    """
    add_new_component(net, Pipe)

    nr_pipes = len(from_junctions)
    index = _get_multiple_index_with_check(net, "pipe", index, nr_pipes)
    _check_branches(net, from_junctions, to_junctions, "pipe")
    _check_std_type(net, std_type, "pipe", "create_pipes")

    pipe_parameters = load_std_type(net, std_type, "pipe")
    entries = {"name": names, "from_junction": from_junctions, "to_junction": to_junctions,
               "std_type": std_type, "length_km": lengths_km,
               "diameter_m": pipe_parameters["inner_diameter_mm"] / 1000, "k_mm": k_mm,
               "loss_coefficient": loss_coefficients, "alpha_w_per_m2k": alpha_w_per_m2k,
               "sections": sections, "in_service": in_service, "type": types, "qext_w": qext_w,
               "text_k": text_k}
    entries.update(kwargs)
    _add_entries_to_table(net, "pipe", index, entries)

    if geodata is not None:
        _add_multiple_branch_geodata(net, "pipe", geodata, index)
    return index


def create_pipes_from_parameters(net, from_junctions, to_junctions, lengths_km, diameters_m, k_mm=1,
                                 loss_coefficients=0, sections=1, alpha_w_per_m2k=0., text_k=293,
                                 qext_w=0., names=None, index=None, geodata=None, in_service=True,
                                 types="pipe", **kwargs):
    """
    Convenience function for creating many pipes at once. Parameters 'from_junctions' and \
    'to_junctions' must be arrays of equal length. Other parameters may be either arrays of the \
    same length or single or values. In any case the line parameters are defined through a single \
    standard type, so all pipes have the same standard type.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junctions: IDs of the junctions on one side which the pipes will be connected to
    :type from_junctions: Iterable(int)
    :param to_junctions: IDs of the junctions on the other side to which the pipes will be \
            connected to
    :type to_junctions: Iterable(int)
    :param lengths_km: Lengths of the pipes in [km]
    :type lengths_km: Iterable or float
    :param diameters_m: The pipe diameters in [m]
    :type diameters_m: Iterable or float
    :param k_mm: Pipe roughness in [mm]
    :type k_mm: Iterable or float, default 1
    :param loss_coefficients: Additional pressure loss coefficients, introduced by e.g. bends
    :type loss_coefficients: Iterable or float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: Iterable or int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficients in [W/(m^2*K)]
    :type alpha_w_per_m2k: Iterable or float, default 0
    :param text_k: Ambient temperatures of pipes in [K]
    :type text_k: Iterable or float, default 293
    :param qext_w: External heat feed-in to the pipes in [W]
    :type qext_w: Iterable or float, default 0
    :param names: Name tags for these pipes
    :type names: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param geodata: The coordinates of the pipes. The first row should be the coordinates of\
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe.
    :type geodata: array, shape=(no_pipes,2L) or (,2L), default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable or bool, default True
    :param types: Identifiers for special types of pipes (e.g. below or above ground)
    :type types: Iterable or str, default "pipe"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> pipe_indices = create_pipes_from_parameters(\
                net, from_junctions=[0, 2, 6], to_junctions=[1, 3, 7], lengths_km=[0.2, 1, 0.3],\
                diameters_m=40e-3)

    """
    add_new_component(net, Pipe)

    index = _get_multiple_index_with_check(net, "pipe", index, len(from_junctions))
    _check_branches(net, from_junctions, to_junctions, "pipe")

    entries = {"name": names, "from_junction": from_junctions, "to_junction": to_junctions,
               "std_type": None, "length_km": lengths_km, "diameter_m": diameters_m, "k_mm": k_mm,
               "loss_coefficient": loss_coefficients, "alpha_w_per_m2k": alpha_w_per_m2k,
               "sections": sections, "in_service": in_service, "type": types, "qext_w": qext_w,
               "text_k": text_k}
    entries.update(kwargs)
    _add_entries_to_table(net, "pipe", index, entries)

    if geodata is not None:
        _add_multiple_branch_geodata(net, "pipe", geodata, index)
    return index


def create_valves(net, from_junctions, to_junctions, diameters_m, opened=True, loss_coefficients=0,
                  names=None, index=None, types='valve', **kwargs):
    """
    Creates a valve element in net["valve"] from valve parameters.

    :param net: The net for which this valve should be created
    :type net: pandapipesNet
    :param from_junctions: IDs of the junctions on one side which the valves will be connected to
    :type from_junctions: Iterable(int)
    :param to_junctions: IDs of the junctions on the other side to which the valves will be \
            connected to
    :type to_junctions: Iterable(int)
    :param diameters_m: The valve diameters in [m]
    :type diameters_m: Iterable or float
    :param opened: Flag to show if the valves are opened and allow for fluid flow or if they are\
            closed to block the fluid flow.
    :type opened: Iterable or bool, default True
    :param loss_coefficients: The pressure loss coefficients introduced by the valve shapes
    :type loss_coefficients: Iterable or float, default 0
    :param names: Name tags for the valves
    :type names: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param types: Identifiers for special types of valves (e.g. below or above ground)
    :type types: Iterable or str, default "valve"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["valve"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> create_valves(net, from_junctions=[0, 1, 4], to_junctions=[1, 5, 6], \
                opened=[True, False, True], diameters_m=4e-3, names=["valve_%d" for d in range(3)])

    """
    add_new_component(net, Valve)

    index = _get_multiple_index_with_check(net, "pipe", index, len(from_junctions))
    _check_branches(net, from_junctions, to_junctions, "pipe")

    entries = {"name": names, "from_junction": from_junctions, "to_junction": to_junctions,
               "diameter_m": diameters_m, "opened": opened, "loss_coefficient": loss_coefficients,
               "type": types}
    entries.update(kwargs)
    _add_entries_to_table(net, "valve", index, entries)

    return index


def create_fluid_from_lib(net, name, overwrite=True):
    """
    Creates a fluid from library (if there is an entry) and sets net["fluid"] to this value.
    Currently existing fluids in the library are: "hgas", "lgas", "hydrogen", "water", "air".

    :param net: The net for which this fluid should be created
    :type net: pandapipesNet
    :param name: The name of the fluid that shall be extracted from the fluid lib
    :type name: str
    :param overwrite: Flag if a possibly existing fluid in the net shall be overwritten
    :type overwrite: bool, default True
    :return: No output

    :Example:
        >>> create_fluid_from_lib(net, name="water")

    """
    _add_fluid_to_net(net, call_lib(name), overwrite=overwrite)


def _get_multiple_index_with_check(net, table, index, number):
    if index is None:
        bid = get_free_id(net[table])
        return np.arange(bid, bid + number, 1)
    if np.any(np.isin(index, net[table].index.values)):
        raise UserWarning("%ss with the ids %s already exist."
                          % (table.capitalize(),
                             net[table].index.values[np.isin(net[table].index.values, index)]))
    return index


def _check_node_elements(net, junctions):
    if np.any(~np.isin(junctions, net["junction"].index.values)):
        junction_not_exist = set(junctions) - set(net["junction"].index.values)
        raise UserWarning("Cannot attach to junctions %s, they do not exist" % junction_not_exist)


def _check_branches(net, from_junctions, to_junctions, table):
    all_junctions = np.array(list(from_junctions) + list(to_junctions))
    if np.any(~np.isin(all_junctions, net.junction.index.values)):
        junction_not_exist = set(all_junctions) - set(net.junction.index)
        raise UserWarning("%s trying to attach to non existing junctions %s"
                          % (table.capitalize(), junction_not_exist))


def _check_std_type(net, std_type, table, function_name):
    if 'std_type' not in net:
        raise UserWarning('%s is defined as std_type in %s but there are no std_types '
                          'defined in your net. You need to define a std_type first or set '
                          'add_stdtypes=True in create_empty_network.' % (std_type, function_name))
    if std_type not in net['std_type'][table]:
        raise UserWarning('%s is not given in std_type (%s). Either change std_type or define new '
                          'one' % (std_type, table))


def _add_entries_to_table(net, table, index, entries, preserve_dtypes=True):
    dtypes = None
    if preserve_dtypes:
        # store dtypes
        dtypes = net[table].dtypes

    dd = pd.DataFrame(index=index, columns=net[table].columns)
    dd = dd.assign(**entries)

    # extend the table by the frame we just created
    if version.parse(pd.__version__) >= version.parse("0.23"):
        net[table] = net[table].append(dd, sort=False)
    else:
        # prior to pandas 0.23 there was no explicit parameter (instead it was standard behavior)
        net[table] = net[table].append(dd)

    # and preserve dtypes
    if preserve_dtypes:
        _preserve_dtypes(net[table], dtypes)


def _add_multiple_branch_geodata(net, table, geodata, index):
    geo_table = "%s_geodata" % table
    dtypes = net[geo_table].dtypes
    df = pd.DataFrame(index=index, columns=net[geo_table].columns)
    # works with single or multiple lists of coordinates
    if len(geodata[0]) == 2 and not hasattr(geodata[0][0], "__iter__"):
        # geodata is a single list of coordinates
        df["coords"] = [geodata] * len(index)
    else:
        # geodata is multiple lists of coordinates
        df["coords"] = geodata

    if version.parse(pd.__version__) >= version.parse("0.23"):
        net[geo_table] = net[geo_table].append(df, sort=False)
    else:
        # prior to pandas 0.23 there was no explicit parameter (instead it was standard behavior)
        net[geo_table] = net[geo_table].append(df)

    _preserve_dtypes(net[geo_table], dtypes)
