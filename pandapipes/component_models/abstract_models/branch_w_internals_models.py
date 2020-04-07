# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.constants import NORMAL_PRESSURE, GRAVITATION_CONSTANT, NORMAL_TEMPERATURE, \
    P_CONVERSION
from pandapipes.idx_branch import FROM_NODE, TO_NODE, LENGTH, D, TINIT, AREA, K, RHO, ETA, \
    VINIT, RE, LAMBDA, LOAD_VEC_NODES, LOSS_COEFFICIENT as LC, T_OUT, CP, PL, JAC_DERIV_DP, \
    JAC_DERIV_DP1, JAC_DERIV_DV, \
    JAC_DERIV_DV_NODE, \
    LOAD_VEC_BRANCHES, ELEMENT_IDX, ACTIVE
from pandapipes.idx_node import TINIT as TINIT_NODE, L, node_cols, PINIT, HEIGHT, PAMB
from pandapipes.pipeflow_setup import add_table_lookup, get_lookup, get_table_number
from pandapipes.properties.fluids import get_fluid
from pandapipes.component_models.auxiliaries.derivative_toolbox import calc_der_lambda, calc_lambda

try:
    from numba import jit
except ImportError:
    from pandapower.pf.no_numba import jit

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWInternalsComponent(BranchComponent):
    """

    """

    @classmethod
    def internal_node_name(cls):
        return NotImplementedError

    @classmethod
    def create_node_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start,
                            current_table, internal_nodes_lookup):
        """
        Function which creates node lookups.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param ft_lookups:
        :type ft_lookups:
        :param table_lookup:
        :type table_lookup:
        :param idx_lookups:
        :type idx_lookups:
        :param current_start:
        :type current_start:
        :param current_table:
        :type current_table:
        :param internal_nodes_lookup:
        :type internal_nodes_lookup:
        :return:
        :rtype:
        """
        internal_nodes = cls.get_internal_pipe_number(net) - 1
        end = current_start
        ft_lookups[cls.table_name()] = None
        if np.any(internal_nodes > 0):
            int_nodes_num = int(np.sum(internal_nodes))
            internal_pipes = internal_nodes + 1
            int_pipes_num = int(np.sum(internal_pipes))
            end = current_start + int_nodes_num
            add_table_lookup(table_lookup, cls.internal_node_name(), current_table)
            ft_lookups[cls.internal_node_name()] = (current_start, end)
            return end, current_table + 1, internal_nodes, internal_pipes, int_nodes_num, int_pipes_num
        else:
            return end, current_table + 1, 0, 0, 0, 0

    @classmethod
    def create_branch_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_table,
                              current_start):
        """
        Function which creates branch lookups.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param ft_lookups:
        :type ft_lookups:
        :param table_lookup:
        :type table_lookup:
        :param idx_lookups:
        :type idx_lookups:
        :param current_table:
        :type current_table:
        :param current_start:
        :type current_start:
        :return:
        :rtype:
        """
        end = current_start + int(np.sum(cls.get_internal_pipe_number(net)))
        ft_lookups[cls.table_name()] = (current_start, end)
        add_table_lookup(table_lookup, cls.table_name(), current_table)
        return end, current_table + 1

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        table_lookup = get_lookup(net, "node", "table")
        table_nr = get_table_number(table_lookup, cls.internal_node_name())
        if table_nr is None:
            return None, 0, 0, None, None, None
        ft_lookup = get_lookup(net, "node", "from_to")
        f, t = ft_lookup[cls.internal_node_name()]

        int_node_pit = node_pit[f:t, :]
        int_node_pit[:, :] = np.array([table_nr, 0, L] + [0] * (node_cols - 3))
        int_node_number = cls.get_internal_pipe_number(net) - 1

        int_node_pit[:, ELEMENT_IDX] = np.arange(t - f)

        f_junction, t_junction = ft_lookup[node_name]
        junction_pit = node_pit[f_junction:t_junction, :]
        from_junctions = net[cls.table_name()].from_junction.values.astype(np.int32)
        to_junctions = net[cls.table_name()].to_junction.values.astype(np.int32)
        return table_nr, int_node_number, int_node_pit, junction_pit, from_junctions, to_junctions

    @classmethod
    def create_pit_branch_entries(cls, net, branch_winternals_pit, node_name):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        branch_winternals_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_winternals_pit, node_name)

        internal_pipe_number = cls.get_internal_pipe_number(net)
        node_ft_lookups = get_lookup(net, "node", "from_to")

        if cls.internal_node_name() in node_ft_lookups:
            pipe_nodes_from, pipe_nodes_to = node_ft_lookups[cls.internal_node_name()]
            pipe_nodes_idx = np.arange(pipe_nodes_from, pipe_nodes_to)
            insert_places = np.repeat(np.arange(len(from_nodes)), internal_pipe_number - 1)
            from_nodes = np.insert(from_nodes, insert_places + 1, pipe_nodes_idx)
            to_nodes = np.insert(to_nodes, insert_places, pipe_nodes_idx)

        branch_winternals_pit[:, ELEMENT_IDX] = np.repeat(net[cls.table_name()].index.values,
                                                          internal_pipe_number)
        branch_winternals_pit[:, FROM_NODE] = from_nodes
        branch_winternals_pit[:, TO_NODE] = to_nodes
        branch_winternals_pit[:, TINIT] = (node_pit[from_nodes, TINIT_NODE] + node_pit[
            to_nodes, TINIT_NODE]) / 2
        fluid = get_fluid(net)
        branch_winternals_pit[:, RHO] = fluid.get_density(branch_winternals_pit[:, TINIT])
        branch_winternals_pit[:, ETA] = fluid.get_viscosity(branch_winternals_pit[:, TINIT])
        branch_winternals_pit[:, CP] = fluid.get_heat_capacity(branch_winternals_pit[:, TINIT])
        branch_winternals_pit[:, ACTIVE] = \
            np.repeat(net[cls.table_name()][cls.active_identifier()].values,internal_pipe_number)

        return branch_winternals_pit, internal_pipe_number

    @classmethod
    def get_internal_pipe_number(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """
        return net[cls.table_name()].sections.values

    @classmethod
    def get_internal_results(cls, net, branch):
        """

        :param net:
        :type net:
        :param pipe:
        :type pipe:
        :return:
        :rtype:
        """
        raise NotImplementedError

class PipeComponentFrank(BranchWInternalsComponent):
    def calculate_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates derivatives. Special version for CO2-pipelines with pressure
        dependent densities from database.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_component_pit:
        :type branch_component_pit:
        :param node_pit:
        :type node_pit:
        :param idx_lookups:
        :type idx_lookups:
        :param options:
        :type options:
        :return: No Output.
        """
        f, t = idx_lookups[cls.table_name()]
        branch_component_pit = branch_pit[f:t, :]
        if branch_component_pit.size == 0:
            return
        fluid = get_fluid(net)
        gas_mode = fluid.is_gas
        friction_model = options["friction_model"]
        g_const = GRAVITATION_CONSTANT

        from_nodes = branch_component_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = branch_component_pit[:, TO_NODE].astype(np.int32)
        p_init_i = node_pit[from_nodes, PINIT]
        p_init_i1 = node_pit[to_nodes, PINIT]
        p_init_i_abs = p_init_i + node_pit[from_nodes, PAMB]
        p_init_i1_abs = p_init_i1 + node_pit[to_nodes, PAMB]
        height_difference = node_pit[from_nodes, HEIGHT] - node_pit[to_nodes, HEIGHT]
        length = branch_component_pit[:, LENGTH]
        dummy = length != 0

        # TODO: calc mean pressure
        p_m = np.empty_like(p_init_i_abs)
        mask = p_init_i_abs != p_init_i1_abs
        p_m[~mask] = p_init_i_abs[~mask]
        p_m[mask] = 2 / 3 * (p_init_i_abs[mask] ** 3 - p_init_i1_abs[mask] ** 3) \
                    / (p_init_i_abs[mask] ** 2 - p_init_i1_abs[mask] ** 2)

        # temperature like in calculate_derivatives_thermal
        t_init_i = node_pit[from_nodes, TINIT_NODE]
        t_init_i1 = branch_component_pit[:, T_OUT]
        t_m = (t_init_i1 + t_init_i) / 2

        # rho = branch_component_pit[:, RHO] # TODO: Abhängigkeit von p
        rho = fluid.get_property("density", p_m, t_m)
        # eta = branch_component_pit[:, ETA] # TODO: Abhängigkeit von p
        eta = fluid.get_property("viscosity", p_m, t_m)
        d = branch_component_pit[:, D]
        k = branch_component_pit[:, K]


        loss_coef = branch_component_pit[:, LC]
        t_init = (node_pit[from_nodes, TINIT_NODE] + node_pit[to_nodes, TINIT_NODE]) / 2
        branch_component_pit[:, TINIT] = t_init
        v_init = branch_component_pit[:, VINIT]

        v_init2 = v_init * np.abs(v_init)

        lambda_pipe, re = calc_lambda(v_init, eta, rho, d, k, gas_mode, friction_model, dummy)
        der_lambda_pipe = calc_der_lambda(v_init, eta, rho, d, k, friction_model, lambda_pipe)
        branch_component_pit[:, RE] = re
        branch_component_pit[:, LAMBDA] = lambda_pipe
        cls.calculate_pressure_lift(net, branch_component_pit, node_pit)
        pl = branch_component_pit[:, PL]

        if not gas_mode:
            branch_component_pit[:, JAC_DERIV_DV] = \
                rho / (P_CONVERSION * 2) * (length / d * (der_lambda_pipe * v_init2 + 2 *
                lambda_pipe * np.abs(v_init)) + 2 * loss_coef * np.abs(v_init))

            branch_component_pit[:, LOAD_VEC_BRANCHES] = \
                - (-p_init_i_abs + p_init_i1_abs - pl
                   - rho * g_const * height_difference / P_CONVERSION
                   + (length * lambda_pipe / d + loss_coef) / (P_CONVERSION * 2) * rho * v_init2)

            branch_component_pit[:, JAC_DERIV_DP] = -1
            branch_component_pit[:, JAC_DERIV_DP1] = 1
        else:
            # Formulas for gas pressure loss according to laminar version described in STANET 10
            # manual, page 1623

            # compressibility settings

            comp_fact = get_fluid(net).get_property("compressibility", p_m)

            const_lambda = NORMAL_PRESSURE * rho * comp_fact * t_init \
                           / (NORMAL_TEMPERATURE * P_CONVERSION)
            const_height = rho * NORMAL_TEMPERATURE / (2 * NORMAL_PRESSURE * t_init * P_CONVERSION)

            branch_component_pit[:, LOAD_VEC_BRANCHES] = \
                -(-p_init_i_abs + p_init_i1_abs - pl + const_lambda * v_init2 * (
                            lambda_pipe * length / d + loss_coef)
                  * (p_init_i_abs + p_init_i1_abs) ** (-1)
                  - const_height * (p_init_i_abs + p_init_i1_abs) * g_const * height_difference)

            branch_component_pit[:, JAC_DERIV_DP] = \
                -1. - const_lambda * v_init2 * (lambda_pipe * length / d + loss_coef) \
                * (p_init_i_abs + p_init_i1_abs) ** (-2) \
                - const_height * g_const * height_difference

            branch_component_pit[:, JAC_DERIV_DP1] = \
                1. - const_lambda * v_init2 * (lambda_pipe * length / d + loss_coef) \
                * (p_init_i_abs + p_init_i1_abs) ** (-2) \
                - const_height * g_const * height_difference

            branch_component_pit[:, JAC_DERIV_DV] = \
                2 * const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) \
                * np.abs(v_init) * lambda_pipe * length / d \
                + const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) * v_init2 \
                * der_lambda_pipe * length / d \
                + 2 * const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) * np.abs(v_init) \
                * loss_coef

        mass_flow_dv = rho * branch_component_pit[:, AREA]
        branch_component_pit[:, JAC_DERIV_DV_NODE] = mass_flow_dv
        branch_component_pit[:, LOAD_VEC_NODES] = mass_flow_dv * v_init

    # TODO: modify to implement density dependency of pressure
    def extract_results(cls, net, options, node_name):
        results = super().extract_results(net, options, node_name)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        fa, ta = get_lookup(net, "branch", "from_to_active")[cls.table_name()]

        placement_table = np.argsort(net[cls.table_name()].index.values)
        idx_pit = net["_pit"]["branch"][f:t, ELEMENT_IDX]
        pipe_considered = get_lookup(net, "branch", "active")[f:t]
        idx_sort, active_pipes, internal_pipes = _sum_by_group(
            idx_pit, pipe_considered.astype(np.int32), np.ones_like(idx_pit, dtype=np.int32))
        active_pipes = active_pipes > 0.99
        placement_table = placement_table[active_pipes]
        branch_pit = net["_active_pit"]["branch"][fa:ta, :]

        return placement_table, branch_pit, results