# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_w_internals_models import BranchWInternalsComponent
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
from pandapipes.internals_toolbox import _sum_by_group



class PipeComponent2D(BranchWInternalsComponent):

    @classmethod
    def calculate_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates derivatives. Special version for pipelines with pressure
        dependent densities from database, e.g. for CO2.

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

        p_m = np.empty_like(p_init_i_abs)
        mask = p_init_i_abs != p_init_i1_abs
        p_m[~mask] = p_init_i_abs[~mask]
        p_m[mask] = 2 / 3 * (p_init_i_abs[mask] ** 3 - p_init_i1_abs[mask] ** 3) \
                    / (p_init_i_abs[mask] ** 2 - p_init_i1_abs[mask] ** 2)
        t_init = (node_pit[from_nodes, TINIT_NODE] + node_pit[to_nodes, TINIT_NODE]) / 2

        rho = fluid.get_property("density", t_init, p_m)
        eta = fluid.get_property("viscosity", t_init, p_m)
        d = branch_component_pit[:, D]
        k = branch_component_pit[:, K]

        loss_coef = branch_component_pit[:, LC]

        branch_component_pit[:, TINIT] = t_init
        v_init = branch_component_pit[:, VINIT]

        v_init2 = v_init * np.abs(v_init)

        lambda_pipe, re = calc_lambda(v_init, eta, rho, d, k, gas_mode, friction_model, dummy)
        der_lambda_pipe = calc_der_lambda(v_init, eta, rho, d, k, friction_model, lambda_pipe)
        branch_component_pit[:, RE] = re
        branch_component_pit[:, LAMBDA] = lambda_pipe
        cls.calculate_pressure_lift(net, branch_component_pit, node_pit)
        pl = branch_component_pit[:, PL]

# TODO: which mode to choose for CO2 if rho(p,T) and eta(p,T) are already refreshed from database?
#         if not gas_mode:
        if True: # # caution, needs validation -  maybe we need parts from the else clause
            branch_component_pit[:, JAC_DERIV_DV] = \
                rho / (P_CONVERSION * 2) * (length / d * (der_lambda_pipe * v_init2 + 2 *
                lambda_pipe * np.abs(v_init)) + 2 * loss_coef * np.abs(v_init))

            branch_component_pit[:, LOAD_VEC_BRANCHES] = \
                - (-p_init_i_abs + p_init_i1_abs - pl
                   - rho * g_const * height_difference / P_CONVERSION
                   + (length * lambda_pipe / d + loss_coef) / (P_CONVERSION * 2) * rho * v_init2)

            branch_component_pit[:, JAC_DERIV_DP] = -1
            branch_component_pit[:, JAC_DERIV_DP1] = 1
        # else:
        #     # Formulas for gas pressure loss according to laminar version described in STANET 10
        #     # manual, page 1623
        #
        #     # compressibility settings
        #
        #     # TODO: is compressibility still required if rho has been updated already?
        #     comp_fact = get_fluid(net).get_property("compressibility", p_m)
        #
        #     const_lambda = NORMAL_PRESSURE * rho * comp_fact * t_init \
        #                    / (NORMAL_TEMPERATURE * P_CONVERSION)
        #     const_height = rho * NORMAL_TEMPERATURE / (2 * NORMAL_PRESSURE * t_init * P_CONVERSION)
        #
        #     branch_component_pit[:, LOAD_VEC_BRANCHES] = \
        #         -(-p_init_i_abs + p_init_i1_abs - pl + const_lambda * v_init2 * (
        #                     lambda_pipe * length / d + loss_coef)
        #           * (p_init_i_abs + p_init_i1_abs) ** (-1)
        #           - const_height * (p_init_i_abs + p_init_i1_abs) * g_const * height_difference)
        #
        #     branch_component_pit[:, JAC_DERIV_DP] = \
        #         -1. - const_lambda * v_init2 * (lambda_pipe * length / d + loss_coef) \
        #         * (p_init_i_abs + p_init_i1_abs) ** (-2) \
        #         - const_height * g_const * height_difference
        #
        #     branch_component_pit[:, JAC_DERIV_DP1] = \
        #         1. - const_lambda * v_init2 * (lambda_pipe * length / d + loss_coef) \
        #         * (p_init_i_abs + p_init_i1_abs) ** (-2) \
        #         - const_height * g_const * height_difference
        #
        #     branch_component_pit[:, JAC_DERIV_DV] = \
        #         2 * const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) \
        #         * np.abs(v_init) * lambda_pipe * length / d \
        #         + const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) * v_init2 \
        #         * der_lambda_pipe * length / d \
        #         + 2 * const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) * np.abs(v_init) \
        #         * loss_coef

        mass_flow_dv = rho * branch_component_pit[:, AREA]
        branch_component_pit[:, JAC_DERIV_DV_NODE] = mass_flow_dv
        branch_component_pit[:, LOAD_VEC_NODES] = mass_flow_dv * v_init
