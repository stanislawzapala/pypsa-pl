import logging
import multiprocessing as mp

from pypsa_pl.helper_functions import ignore_future_warnings
from pypsa_pl.custom_constraints import (
    define_operational_limit_link,
    define_operational_limit_per_area,
    define_custom_primary_energy_limit,
    define_nominal_constraints_per_area_carrier,
    define_annual_capacity_utilisation_constraints,
    define_parent_children_capacity_constraints,
    define_fixed_ratios_to_space_heating,
    define_heating_capacity_utilisation_constraints,
    define_centralised_heating_share_constraint,
    define_non_heating_capacity_utilisation_constraints,
    define_minimum_synchronous_generation,
    remove_annual_carrier_demand_constraints,
    define_proportional_expansion_constraint,
    define_minimum_bev_charge_level_constraint,
)
from pypsa_pl.config import project_dir

# Set the number of threads to be used by the solver to 75% of available cores, but never more than 8
n_threads = min(max(1, int(mp.cpu_count() * 0.75 + 0.01)), 8)


solver_options = lambda eps=1e-6, log_dir=project_dir, flags=[]: {
    "highs": {
        "threads": n_threads,
        "solver": "ipm",
        "run_crossover": "off",
        "small_matrix_value": 1e-7,
        "large_matrix_value": 1e12,
        "ipm_optimality_tolerance": eps,
        "dual_feasibility_tolerance": eps,
        "primal_feasibility_tolerance": eps * 10,
        "parallel": "on",
        "random_seed": 0,
        **{flag: "on" for flag in flags},
    },
    "gurobi": {
        "Threads": n_threads,
        "Method": 2,  # barrier (IPM)
        "Crossover": 0,
        # "OptimalityTol": eps,  # only relevent if crossover is on
        "BarConvTol": eps,
        "FeasibilityTol": eps * 10,
        # Those help to stabilise the solver in the final iterations
        "NumericFocus": 1,
        "BarCorrectors": 1,
        # "BarHomogeneous": 1, # this sometimes makes the final iterations more difficult to converge
        # ***
        # Those help reduce memory consumption in presolve step & speeds up the whole optimisation
        "AggFill": 0,
        "PreDual": 0,
        # ***
        "Seed": 0,
        "LogFile": str(log_dir("solver.log")),
        **{flag: 1 for flag in flags},
    },
    "mosek": {
        "MSK_IPAR_NUM_THREADS": n_threads,
        # "MSK_IPAR_PRESOLVE_USE": "MSK_PRESOLVE_MODE_OFF",
        "MSK_IPAR_OPTIMIZER": "MSK_OPTIMIZER_INTPNT",
        "MSK_DPAR_INTPNT_TOL_PFEAS": eps * 10,
        "MSK_DPAR_INTPNT_TOL_DFEAS": eps,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": eps,
        "MSK_DPAR_INTPNT_TOL_INFEAS": eps / 10,
        **{flag: 1 for flag in flags},
    },
}


def add_epsilon_to_optimal_capacities(network, eps_rel=5e-5, eps_abs=1e-2):
    # Virtual capacities should not be increased as this would increase the annual flows
    virtual_capacities = [
        "centralised space heating",
        "centralised water heating",
        "centralised other heating",
        "decentralised space heating",
        "decentralised water heating",
        "light vehicle mobility",
        "hydrogen",
    ]

    # A negligible but non-zero capacity increase to avoid infeasibility in dispatch-only optimisation
    for component, nom_attr in [
        ("Generator", "p_nom"),
        ("Link", "p_nom"),
        ("Store", "e_nom"),
    ]:
        is_not_virtual = ~network.static(component)["carrier"].isin(virtual_capacities)
        is_extendable = network.static(component)[f"{nom_attr}_extendable"]
        network.static(component).loc[
            is_extendable & is_not_virtual, f"{nom_attr}_opt"
        ] = (
            network.static(component).loc[
                is_extendable & is_not_virtual, f"{nom_attr}_opt"
            ]
            * (1 + eps_rel)
            + eps_abs
        )


def reset_capacities(network, params):
    for component, nom_attr in [
        ("Generator", "p_nom"),
        ("Link", "p_nom"),
        ("Store", "e_nom"),
    ]:
        is_to_invest = network.static(component)["technology"].isin(
            params["investment_technologies"]
        )
        is_to_retire = network.static(component)["technology"].isin(
            params["retirement_technologies"]
        )
        network.static(component)[f"{nom_attr}_extendable"] = (
            is_to_invest | is_to_retire
        )

        is_cumulative = network.static(component)["lifetime"] == 1
        is_active = network.static(component)["build_year"] == params["year"]

        is_to_invest &= ~is_cumulative & is_active
        network.static(component).loc[is_to_invest, nom_attr] = network.static(
            component
        ).loc[is_to_invest, f"{nom_attr}_min"]
        is_to_retire &= is_cumulative & is_active
        network.static(component).loc[is_to_retire, nom_attr] = network.static(
            component
        ).loc[is_to_retire, f"{nom_attr}_max"]


@ignore_future_warnings
def create_and_solve_model(network, params, log_dir, fixed_virtual_capacities=False):

    # Turn single index snapshots into multi-index snapshots with period equal to simulation year
    # This is only necessary if we need max_growth constraint
    network.set_investment_periods([params["year"]])

    network.optimize.create_model(
        multi_investment_periods=True,
        transmission_losses=0,
        linearized_unit_commitment=False,
    )

    define_operational_limit_link(network, network.snapshots)
    define_operational_limit_per_area(network, network.snapshots)
    define_custom_primary_energy_limit(network, network.snapshots)
    define_annual_capacity_utilisation_constraints(network, network.snapshots)
    define_minimum_synchronous_generation(network, network.snapshots)

    define_nominal_constraints_per_area_carrier(network, network.snapshots)
    define_parent_children_capacity_constraints(network, network.snapshots)
    define_proportional_expansion_constraint(network, network.snapshots)

    minimum_bev_charge_level = network.meta.get("minimum_bev_charge_level", 0.0)
    if minimum_bev_charge_level > 0.0:
        define_minimum_bev_charge_level_constraint(network, network.snapshots)

    if not fixed_virtual_capacities:
        # These constraints might lead to infesibility in dispatch-only optimisation
        define_fixed_ratios_to_space_heating(network, network.snapshots)
        define_heating_capacity_utilisation_constraints(network, network.snapshots)
        define_non_heating_capacity_utilisation_constraints(network, network.snapshots)
        define_centralised_heating_share_constraint(network, network.snapshots)
    else:
        # Those are the carriers that have associated fixed virtual capacities which determine annual flows
        # These annual flow constraints that might lead to infeasibility in dispatch-only optimisation
        # Hydrogen has to be excluded from 2040 as it has other uses than final use
        # TODO: avoid hard coding year for hydrogen
        remove_annual_carrier_demand_constraints(
            network,
            carriers=[
                "space heating",
                "water heating",
                "other heating",
                "light vehicle mobility",
            ]
            + (["hydrogen"] if params["year"] < 2040 else []),
        )

    eps = params.get("solver_tolerance", 1e-6)
    if fixed_virtual_capacities:
        eps *= 10
    extra_flags = params.get("solver_extra_flags", [])
    solver = params["solver"]

    status, condition = network.optimize.solve_model(
        solver_name=solver,
        solver_options=solver_options(eps, log_dir, extra_flags).get(solver, {}),
    )
    network.meta["solver_status"] = f"{status}: {condition}"

    # Close the model and free up memory
    # if solver == "gurobi":
    #     sm = network.model.solver_model
    #     sm.dispose()


def optimise_network(network, params, log_dir):

    # Workaround for foreign capacities and constraints
    for component in ["Bus", "Generator", "Link", "Store", "GlobalConstraint"]:
        is_not_domestic = ~network.static(component)["area"].str.startswith("PL")
        if component == "Link":
            is_not_domestic &= ~network.static(component)["area_from"].str.startswith(
                "PL"
            )
        carrier_col = (
            "carrier" if component != "GlobalConstraint" else "carrier_attribute"
        )
        network.static(component).loc[is_not_domestic, carrier_col] += (
            " " + network.static(component).loc[is_not_domestic, "area"]
        )

    create_and_solve_model(network, params, log_dir)
    if params["reoptimise_with_fixed_capacities"]:
        # If snapshots=None, maximum recursion error appears
        network_non_fixed = network.copy(snapshots=network.snapshots)
        logging.info("Repeating optimization with optimal capacities fixed...")
        add_epsilon_to_optimal_capacities(network)
        network.optimize.fix_optimal_capacities()
        create_and_solve_model(network, params, log_dir, fixed_virtual_capacities=True)
        reset_capacities(network, params)

    networks = (
        [network_non_fixed] if params["reoptimise_with_fixed_capacities"] else []
    ) + [network]

    for n in networks:

        # At the end, for foreign buses keep area code as part of bus carrier
        for component in ["Generator", "Link", "Store", "GlobalConstraint"]:
            is_not_domestic = ~n.static(component)["area"].str.startswith("PL")
            if component == "Link":
                is_not_domestic &= ~n.static(component)["area_from"].str.startswith(
                    "PL"
                )
            carrier_col = (
                "carrier" if component != "GlobalConstraint" else "carrier_attribute"
            )
            assert (
                n.static(component).loc[is_not_domestic, "area"].str.len() == 2
            ).all()
            n.static(component).loc[is_not_domestic, carrier_col] = (
                n.static(component).loc[is_not_domestic, carrier_col].str[:-3]
            )

    return networks
