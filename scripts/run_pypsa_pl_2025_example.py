import sys
import gc
import logging
import numpy as np
import pandas as pd

from pypsa_pl.config import data_dir
from pypsa_pl.build_network import load_network
from pypsa_pl.run_simulation import run_simulation, generate_outputs

# Warning: optimisation with fixed capacities often fails, check the results for marginal prices carefully

# Extra assumptions (global)
dsr_as_final_use_share = 0.1
dsr_as_hmv_final_use_share = 0.15
interconnector_factor = 0.5
grid_loss = 0.07  # model now endogenously determines grid losses, so this in an approximate value
cdr_factor = {
    2020: 0,
    2024: 0,
    2025: 0,
    2030: 0,
    2035: 1,
    2040: 1,
    2045: 1,
    2050: 1,
}
neighbour_electricity_factor = 0.85 * (1 - grid_loss)
neighbour_hydrogen_factor = 0.75
neighbour_dsr_factor = 2.5
last_historical_year = 2024
grid_cost_reduction_factor = 0.01  # only for (virtually) zero grid cost runs


def define_scenarios(delete_capacity_file=False):

    # Define years for scenarios
    years = [2025, 2030, 2035, 2040]

    # Uncomment scenarios to be run
    scenarios = [
        # (0) Main scenario (Instrat Main 2025)
        # "instrat_ambitious+trade",
        # "instrat_ambitious+trade+medium+dunkelflaute",
        "instrat_ambitious+trade+mini",
        # (1) Delayed RES variant (Instrat Less RES 2025)
        # "instrat_ambitious+trade+delayed_RES",
        # "instrat_ambitious+trade+delayed_RES+medium+dunkelflaute",
        # "instrat_ambitious+trade+delayed_RES+mini",
        # (2) Less electrolysis variant (Instrat Less Electrolysis 2025)
        # "instrat_ambitious+trade+less_electrolysis,
        # "instrat_ambitious+trade+less_electrolysis+medium+dunkelflaute",
        # "instrat_ambitious+trade+less_electrolysis+mini",
        # (3) Less heat pumps variant (Instrat Less Heat Pumps 2025)
        # "instrat_ambitious+trade+less_heat_pumps",
        # "instrat_ambitious+trade+less_heat_pumps+medium+dunkelflaute",
        # "instrat_ambitious+trade+less_heat_pumps+mini",
        # (4) Oversized grids variant
        # "instrat_ambitious+trade+zero_grid_cost",
        # "instrat_ambitious+trade+zero_grid_cost+medium+dunkelflaute",
        # "instrat_ambitious+trade+zero_grid_cost+mini",
    ]

    # Switch between copper plate and voivodeship-resolved scenarios
    copper_plate = False
    input_suffix = "_voivodeships"
    if copper_plate:
        scenarios = [f"{scenario}+copperplate" for scenario in scenarios]
        input_suffix = ""

    # Extra input from demand projections
    centralised_heating_shares = pd.read_csv(
        data_dir(
            "clean", "centralised_heating_shares", "centralised_heating_shares.csv"
        ),
        index_col=0,
    )
    centralised_heating_shares = {
        y: centralised_heating_shares[str(y)].to_dict() if y >= 2030 else None
        for y in years
    }

    # logging.info("Centralised heating shares:", centralised_heating_shares)

    p_min_synchronous = {
        2020: 0,  # 6 GW enforced by p_min_pu for power-only capacities + must run CHPs, ~35% of avg. generation (155 TWh -> 6200 MW)
        2024: 0,
        2025: 0,  # 5 GW enforced by p_min_pu for power-only capacities + must run CHPs, ~25% of avg. generation (175 TWh -> 5000 MW)
        2030: 4500,  # 20% of avg. generation (195 TWh -> 4500 MW)
        2035: 2700,  # 10% of avg. generation (240 TWh -> 2700 MW)
        2040: 1800,  # 5% of avg. generation (320 TWh -> 1800 MW)
        2045: 0,
        2050: 0,
    }

    param_dict = {scenario: {} for scenario in scenarios}

    for scenario in scenarios:
        if delete_capacity_file:
            data_dir("input", f"installed_capacity;variant={scenario}.csv").unlink(
                missing_ok=True
            )
        for year in years:
            installed_capacity = (
                [f"historical_totals{input_suffix}"]
                if year == last_historical_year
                else [f"historical+instrat_projection{input_suffix}"]
            )

            annual_energy_flows = (
                [f"historical{input_suffix}"]
                if year == last_historical_year
                else [f"instrat_projection{input_suffix}", "constraints"]
            )
            capacity_utilisation = (
                ["historical"]
                if year == last_historical_year
                else ["instrat_projection"]
            )
            capacity_addition_potentials = [
                "instrat_projection",
                f"instrat_res_potentials{input_suffix}",
            ]
            if not "copperplate" in scenario:
                capacity_addition_potentials += [
                    "instrat_other_potentials_voivodeships"
                ]

            if "trade" in scenario:
                installed_capacity += [f"interconnectors{input_suffix}", "neighbours"]
                annual_energy_flows += ["neighbours"]
                capacity_utilisation += ["neighbours"]

            # In case we model beyond 2025, we have to use the scenario-specific file
            if year > 2025:
                installed_capacity = scenario

            # Assume prosumer self consumption is included in total demand and generation after 2025
            # Historically, the self-consumed part of prosumer PV generation (~20%) has not been not reported
            if year <= last_historical_year:
                pv_self_consumption = 0.2
            else:
                pv_self_consumption = 0

            investment_technologies = [
                # Power grid - include to either determine baseline or to estimate investment needs
                "distribution HMV",
                "transformation HMV-LV",
                "transformation LV-HMV",
                "distribution LV",
                "transformation EHV-HMV",
                "transformation HMV-EHV",
                # vRES connections are set at the baseline
                # "connection HMV-vRES",
                # "connection vRES-HMV"
                # "direct line vRES",
            ]
            retirement_technologies = [
                # Virtual components - always need to be included
                "centralised space heating",
                "centralised water heating",
                "centralised other heating",
                "decentralised space heating",
                "decentralised water heating",
                "light vehicle mobility",
                "hydrogen",
            ]

            if year < 2030:
                # Allow the capacities of ICE vehicles and natural gas boilers to calibrate themselves to total heating and mobility demands
                investment_technologies += ["ICE vehicle", "natural gas boiler"]
                retirement_technologies += ["ICE vehicle", "natural gas boiler"]

            if year >= 2030:
                investment_technologies += [
                    # *** Power and CHP ***
                    "wind onshore",
                    "wind offshore",
                    "solar PV ground",
                    "solar PV ground E",
                    "solar PV ground W",
                    "solar PV roof",
                    "battery large storage",
                    "battery large power",
                    "battery large charger",
                    "hydro PSH power",
                    "hydro PSH pump",
                    "hydro PSH storage",
                    "natural gas power CCGT",
                    "natural gas power peaker",
                    "natural gas CHP CCGT",
                    "biogas production",
                    "biogas upgrading",
                    "biogas CHP",
                    "biomass agriculture CHP",
                    # *** Heating - centralised ***
                    "natural gas heat",
                    "biomass agriculture heat",
                    "heat pump large AW",
                    "resistive heater large",
                    "heat storage large tank",
                    "heat storage large tank discharge",
                    "heat storage large tank charge",
                    # *** Heating - decentralised ***
                    "natural gas boiler",
                    "heat pump small AW",
                    "resistive heater small",
                    "heat storage small",
                    "heat storage small discharge",
                    "heat storage small charge",
                    # *** Mobility ***
                    "ICE vehicle",
                    "BEV",
                    "BEV battery",
                    "BEV charger",
                    # *** Hydrogen ***
                    "natural gas reforming",
                    "hydrogen electrolysis",
                    "hydrogen storage",
                    # *** Power grid ***
                    "connection HMV-vRES",
                    "connection vRES-HMV",
                    "transmission line AC",
                ]
                if "v2g" in scenario:
                    investment_technologies += [
                        "BEV V2G",
                    ]
                retirement_technologies += [
                    # *** Coal-based centralised power and heat technologies ***
                    "hard coal power old",
                    "hard coal power SC",
                    "lignite power old",
                    "lignite power SC",
                    "hard coal CHP",
                    "hard coal heat",
                    # *** Allow changing shares of district heating systems ***
                    "district heating",
                ]
            if year > 2030:
                # New technologies potentially available after 2030
                investment_technologies += [
                    "biomass agriculture CHP CC",
                    "hydrogen power CCGT",
                    "hydrogen power peaker",
                    "hydrogen CHP CCGT",
                    "hydrogen heat",
                ]
                # Technologies forbidden after 2030
                investment_technologies = [
                    tech
                    for tech in investment_technologies
                    if tech
                    not in [
                        "natural gas boiler",  # EU regulations will likely limit new gas boilers after 2030
                        "natural gas power CCGT",  # model will overestimate long-term CF of gas plants, better build peakers
                    ]
                ]
                # However, in less heat pumps variant, keep natural gas boilers
                if "less_heat_pumps" in scenario:
                    investment_technologies += ["natural gas boiler"]

            if year > 2035:
                # New technologies potentially available after 2035
                investment_technologies += [
                    "nuclear power large",
                ]
                # Technologies forbidden after 2035
                investment_technologies = [
                    tech
                    for tech in investment_technologies
                    if tech
                    not in [
                        "ICE vehicle",  # EU policy
                        "natural gas power peaker",  # force switch to hydrogen
                        "natural gas CHP CCGT",  # force switch to hydrogen
                        "natural gas heat",  # force switch to hydrogen
                    ]
                ]

            # Enforce solar PV ground E&W in East West PV variant
            # We do this by forbidding the solar PV ground (S)
            if "east_west_PV" in scenario:
                investment_technologies = [
                    tech
                    for tech in investment_technologies
                    if tech != "solar PV ground"
                ]

            constrained_energy_flows = [
                # Exact demand per voivodeship
                "space heating final use",
                "water heating final use",
                "light vehicle mobility final use",
                # Exact demand per country
                "other heating final use",
                "hydrogen final use",
                # Maximum quantity
                "biomass agriculture supply",
                "biogas substrate supply",
                # Minimum quantities in 2030/2035
                "hydrogen electrolysis",
                "biogas upgrading",
            ]
            # If not copperplate, those carriers do not have specified demand per area
            if not "copperplate" in scenario:
                constrained_energy_flows += [
                    "natural gas final use",
                    "hard coal final use",
                    "biomass wood final use",
                    "other fuel final use",
                    "process emissions final use",
                    "lulucf final use",
                ]

            if year in [2020, last_historical_year]:
                # For 2020 reproduce exact avg. 2019-2021 production
                # For 2024 exact 2024 production
                # Keep biomass power & CHP generation free so it adjusts to remaining demand
                constrained_energy_flows += [
                    "hard coal power",
                    "hard coal CHP",
                    "lignite power",
                    "natural gas CHP",
                    "electricity export",
                    "electricity import",
                    "biogas CHP",
                    "solar PV ground",
                    "solar PV roof",
                ]
                if year == last_historical_year:
                    constrained_energy_flows += [
                        "wind onshore",
                    ]

            timeseries_variant = "full"
            for v in [
                "micro",
                "mini",
                "mini+dunkelflaute",
                "medium",
                "medium+dunkelflaute",
            ]:
                if v in scenario:
                    timeseries_variant = v

            # Default BEV assumptions
            bev_flexibility_factor = 0.5
            bev_minimum_charge_level = 0 # do not apply minimum charge level at 6 am constraint

            param_dict[scenario][year] = {
                # Run name and year
                "run_name": f"pypsa_pl;scenario={scenario};year={year}",
                "year": year,
                # Input data
                "technology_carrier_definitions": "full",
                "technology_cost_data": "instrat_2025",
                "installed_capacity": installed_capacity,
                "annual_energy_flows": annual_energy_flows,
                "capacity_utilisation": capacity_utilisation,
                "capacity_addition_potentials": capacity_addition_potentials,
                "timeseries": timeseries_variant,
                # CO2 emissions
                "co2_emissions": 0 if year == 2050 else None,
                # Weather year
                "weather_year": 2012,
                # Other assumptions
                "discount_rate": 0.045,
                "investment_cost_start_year": 2021,
                "invest_from_zero": True,
                "optimise_industrial_capacities": False,
                "investment_technologies": investment_technologies,
                "retirement_technologies": retirement_technologies,
                "constrained_energy_flows": constrained_energy_flows,
                "reoptimise_with_fixed_capacities": False,  # True
                # CHP behavior
                "fix_public_chp": False,
                "fix_industrial_chp": True,
                "share_space_heating": None,  # irrelevant if fix_public_chp=False
                # Electricity sector
                "prosumer_self_consumption": pv_self_consumption,
                "p_min_synchronous": p_min_synchronous[year],
                "synchronous_carriers": [
                    "hard coal power",
                    "lignite power",
                    "natural gas power",
                    "natural gas power peaker",
                    "biomass wood power",
                    "nuclear power",
                    "hydrogen power",
                    "hard coal CHP",
                    "natural gas CHP",
                    "other CHP",
                    "biomass wood CHP",
                    "biomass agriculture CHP",
                    "biomass agriculture CHP CC",
                    "hydrogen CHP",
                    "biogas CHP",
                    "hydro ROR",
                ],
                "proportional_expansion": [
                    "wind onshore",
                    "wind offshore",
                    "solar PV ground",
                    "solar PV roof",
                    "heat pump small",
                    "BEV",
                ],
                # Heating sector
                "heat_capacity_utilisation": 0.2,
                "centralised_heating_shares": centralised_heating_shares[year],
                # Light vehicle mobility sector
                "light_vehicle_mobility_utilisation": 0.021,
                "bev_flexibility_factor": bev_flexibility_factor,
                "bev_flexibility_max_to_mean_ratio": 1.33,
                "bev_flexible_share": 0.5,  # if exact 0 is desired, set bev_flexibility_factor=0 instead
                "bev_availability_max": 0.9,
                "bev_availability_mean": 0.7,
                "minimum_bev_charge_hour": 6,
                "minimum_bev_charge_level": bev_minimum_charge_level,
                # Technical details - they should not influence numerical results
                "hydrogen_utilisation": 1,
                "space_heating_utilisation": 0.1,
                "water_heating_utilisation": 1,
                "other_heating_utilisation": 1,
                "inf": 999999,
                "reverse_links": True,
                "solver": "gurobi",
                "solver_tolerance": 1e-5,
                "solver_extra_flags": [
                    # "BarHomogenous",
                    # "Crossover",
                ],
            }
    return param_dict


def custom_input_operation(inputs, params):

    # Identify solar PV roof as prosumer electricity source with partially non-reported self-consumption
    def add_qualifier_to_technology(df, technology, qualifier):
        df.loc[df["technology"] == technology, "qualifier"] = qualifier
        return df

    inputs["installed_capacity"] = add_qualifier_to_technology(
        inputs["installed_capacity"],
        "solar PV roof",
        "prosumer",
    )

    # Do not activate p_max_pu_annual constraints
    def remove_p_max_pu_annual_technology_input(df, keep_techs=[]):
        df = df[
            ~(df["parameter"] == "p_max_pu_annual")
            | df["technology"].isin(keep_techs)
            | df["technology"].str.startswith(("wind", "solar"))
        ]
        return df

    inputs["technology_cost_data"] = remove_p_max_pu_annual_technology_input(
        inputs["technology_cost_data"],
        keep_techs=[],
    )

    # Do not model CHP plants as fixed output generators
    def remove_chp_capacity_utilisation_input(df, qualifiers=["public", "industrial"]):
        df = df[
            ~(
                df["technology"].str.contains("CHP")
                & df["qualifier"].isin(qualifiers)
                & df["parameter"].isin(["p_set_pu", "p_set_pu_annual"])
            )
        ]
        return df

    qualifiers = []
    if not params["fix_public_chp"]:
        qualifiers += ["public"]
    if not params["fix_industrial_chp"]:
        qualifiers += ["industrial"]
    inputs["capacity_utilisation"] = remove_chp_capacity_utilisation_input(
        inputs["capacity_utilisation"], qualifiers=qualifiers
    )

    # Remove predefined DSR because we will add custom values for its capacity
    def remove_capacities(
        df, technologies=[], qualifiers=[], areas=[], max_build_year=params["year"]
    ):
        if technologies:
            df = df[~df["technology"].isin(technologies)]
        if qualifiers:
            df = df[~df["qualifier"].isin(qualifiers)]
        if areas:
            df = df[~df["area"].isin(areas)]
        df = df[df["build_year"] <= max_build_year]
        return df

    inputs["installed_capacity"] = remove_capacities(
        inputs["installed_capacity"],
        areas=[],
        technologies=["DSR reduction"],
        qualifiers=[],
    )

    # TODO: if simulating 2050, remove virtual capacities from other bus for heating in 2050
    # If not done, one might encounter infeasibilities

    # Decrease the capacity of interconnectors, to have more realistic electricity trade flows
    def rescale_interconnectors(df, p_max_pu):
        df = df.set_index(["area", "technology", "qualifier", "year", "parameter"])
        for technology in [
            "electricity export AC",
            "electricity import AC",
            "electricity export DC",
            "electricity import DC",
        ]:
            for area in ["PL", "DE", "SK", "CZ", "LT", "SE"]:
                df.loc[area, technology, np.nan, params["year"], "p_max_pu"] = [
                    p_max_pu
                ]
        df = df.reset_index()
        return df

    inputs["capacity_utilisation"] = rescale_interconnectors(
        inputs["capacity_utilisation"],
        p_max_pu=interconnector_factor,
    )

    # For 2020 and 2025 rescale district heating capacities such that they match the heat demand scenario (see data/clean/centralised_heating_shares/assumed_heat_centralised_shares.csv)
    # For 2025 also include building retrofits (equivalent to 430 MW of heat capacity reduction, see installed capacity inputs)
    def rescale_district_heating(df, years_generations):
        is_dh = df["technology"] == "district heating"
        df_dh = df[is_dh].groupby("build_year").agg({"nom": "sum"})

        df_dh["original_generation"] = (
            df_dh["nom"] * params["heat_capacity_utilisation"] * 8760 / 1e6
        )
        df_dh["target_generation"] = df_dh["original_generation"]
        for year, generation in years_generations:
            df_dh.loc[year, "target_generation"] = generation
        df_dh["dh_ratio"] = df_dh["target_generation"] / df_dh["original_generation"]
        df_dh = df_dh.dropna().reset_index()[["build_year", "dh_ratio"]]
        df = df.merge(df_dh, how="left", on="build_year")

        is_dh = df["technology"] == "district heating"
        df.loc[is_dh, "nom"] *= df.loc[is_dh, "dh_ratio"].fillna(1)
        df = df.drop(columns="dh_ratio")

        return df

    inputs["installed_capacity"] = rescale_district_heating(
        inputs["installed_capacity"],
        years_generations=[
            (2020, 72.1),
            (2025, 68.7 - 430 * params["heat_capacity_utilisation"] * 8760 / 1e6),
        ],
    )

    # These capacities are calibrated as retirement technologies, hence the starting point has to be infinity
    def set_extendable_virtual_capacities_to_infinity(df):
        df.loc[
            df["technology"].isin(
                [
                    "centralised space heating",
                    "centralised water heating",
                    "centralised other heating",
                    "decentralised space heating",
                    "decentralised water heating",
                    "light vehicle mobility",
                    "hydrogen",
                    "district heating",
                ]
            )
            & df["technology"].isin(params["retirement_technologies"]),
            "nom",
        ] = np.inf
        return df

    inputs["installed_capacity"] = set_extendable_virtual_capacities_to_infinity(
        inputs["installed_capacity"]
    )

    # One might reduce or turn off V2G potential of BEVs
    # Warning: this should not be run multiple times!
    def rescale_v2g(df_tech, df_cap, factor):
        is_v2g_parent_ratio = (df_tech["technology"] == "BEV V2G") & (
            df_tech["parameter"] == "parent_ratio"
        )
        df_tech.loc[is_v2g_parent_ratio, "value"] = (
            pd.to_numeric(df_tech.loc[is_v2g_parent_ratio, "value"], errors="coerce")
            * factor
        )
        is_v2g_nom = df_cap["technology"] == "BEV V2G"
        df_cap.loc[is_v2g_nom, "nom"] = (
            pd.to_numeric(df_cap.loc[is_v2g_nom, "nom"], errors="coerce") * factor
        )
        return df_tech, df_cap

    v2g_factor = 0.0
    if "v2g" in params["run_name"] and params["year"] > 2025:
        v2g_factor = 0.25
    inputs["technology_cost_data"], inputs["installed_capacity"] = rescale_v2g(
        inputs["technology_cost_data"],
        inputs["installed_capacity"],
        factor=v2g_factor,
    )

    # Set v2g variable cost
    def set_v2g_cost(df_tech, value):
        is_v2g_cost = (df_tech["technology"] == "BEV V2G") & (
            df_tech["parameter"] == "variable_cost"
        )
        df_tech.loc[is_v2g_cost, "value"] = value
        return df_tech

    v2g_cost = 50
    inputs["technology_cost_data"] = set_v2g_cost(
        inputs["technology_cost_data"], v2g_cost
    )

    # Rescale electricity and hydrogen demand in neighbouring countries
    # This is to calibrate the net trade flows of Poland
    def rescale_neighbours_demand(
        df, neighbour_electricity_factor=1, neighbour_hydrogen_factor=1
    ):
        is_neighbour = ~df["area"].str.startswith("PL")
        for carrier, factor in [
            ("electricity final use", neighbour_electricity_factor),
            ("hydrogen final use", neighbour_hydrogen_factor),
        ]:
            df.loc[is_neighbour & (df["carrier"] == carrier), "value"] *= factor
        return df

    inputs["annual_energy_flows"] = rescale_neighbours_demand(
        inputs["annual_energy_flows"],
        neighbour_electricity_factor,
        neighbour_hydrogen_factor,
    )

    # Hydrogen storage at neighbours need to be increased to make model feasible
    # Neighbouring hydrogen systems are modelled primarily to add some flexible demand to neighbouring power systems
    def increase_hydrogen_storage_in_neighbours(
        df, hours=24 * 7, peaker_efficiency=0.4
    ):
        # Ensure that hydrogen storage can provide fuel for a week of generation at full output of hydrogen peakers
        is_foreign_h2_peaker = ~df["area"].str.startswith("PL") & (
            df["technology"] == "hydrogen power peaker"
        )
        df_peaker = df.loc[is_foreign_h2_peaker, ["area", "build_year", "nom"]]
        df_peaker["h2_storage_nom_min"] = (
            df_peaker["nom"] * hours / peaker_efficiency
        ).round(-1)
        df = df.merge(
            df_peaker.drop(columns="nom"), how="left", on=["area", "build_year"]
        )
        is_foreign_h2_storage = ~df["area"].str.startswith("PL") & (
            df["technology"] == "hydrogen storage"
        )
        df.loc[is_foreign_h2_storage, "nom"] = np.maximum(
            df.loc[is_foreign_h2_storage, "nom"],
            df.loc[is_foreign_h2_storage, "h2_storage_nom_min"],
        )
        df = df.drop(columns="h2_storage_nom_min")
        return df

    inputs["installed_capacity"] = increase_hydrogen_storage_in_neighbours(
        inputs["installed_capacity"],
    )

    # Add DSR technology with potential equal to a share of electricity or HMV electricity final use demand
    def add_dsr_technology(
        df_cap,
        df_flow,
        final_use_share=0.1,
        hmv_final_use_share=0.15,
        neighbour_factor=1,
    ):
        is_electricity_final_use = df_flow["carrier"] == "electricity final use"
        is_hmv_final_use = df_flow["carrier"] == "electricity HMV final use"
        df = df_flow[is_electricity_final_use | is_hmv_final_use].copy()
        share = df["carrier"].map(
            {
                "electricity final use": final_use_share,
                "electricity HMV final use": hmv_final_use_share,
            }
        )
        # Convert TWh/year into MW, increasing by grid losses
        df["nom"] = share * df["value"] / 8760 * 1e6 / (1 - grid_loss)
        df["technology"] = "DSR reduction"
        df["build_year"] = df["year"]
        df["retire_year"] = df["year"]
        df["cumulative"] = True
        df["name"] = df["area"] + " " + df["technology"] + " " + df["year"].astype(str)
        # Neighbors might have a different DSR potential
        is_neighbour = ~df["area"].str.startswith("PL")
        df.loc[is_neighbour, "nom"] *= neighbour_factor
        df["nom"] = df["nom"].round(-1)
        df = df[df["nom"] > 0]
        df = df[df_cap.columns.intersection(df.columns)]
        df_cap = pd.concat([df_cap, df])
        logging.info(
            f"Added DSR reduction potential equal to {(final_use_share * 100):.0f}% of average electricity final use load (if no HMV/LV distinction) and {(hmv_final_use_share * 100):.0f}% of average electricity HMV final use load"
        )
        return df_cap

    inputs["installed_capacity"] = add_dsr_technology(
        inputs["installed_capacity"],
        inputs["annual_energy_flows"],
        final_use_share=dsr_as_final_use_share,
        hmv_final_use_share=dsr_as_hmv_final_use_share,
        neighbour_factor=neighbour_dsr_factor,
    )

    def subtract_endogenous_electricity_consumption_and_losses(df_flow, df_cap):
        efficiency = {
            "resistive heater small": 1,
            "heat pump small AW": 3.1,
            "BEV": 0.85 * 0.9,
        }
        utilisation = {
            "resistive heater small": params["heat_capacity_utilisation"],
            "heat pump small AW": params["heat_capacity_utilisation"],
            "BEV": params["light_vehicle_mobility_utilisation"],
        }

        # Identify sectoral electricity demand sources
        df = df_cap.loc[
            (
                (df_cap["technology"] == "resistive heater small")
                & (df_cap["bus_qualifier"] == "resistive heater")
                # There are also resistive heaters supporting heat pumps - we assume they have negligibly small utilisation
            )
            | (df_cap["technology"] == "heat pump small AW")
            | (df_cap["technology"] == "BEV"),
            ["area", "technology", "build_year", "nom"],
        ].rename(columns={"build_year": "year", "nom": "sectoral_consumption"})
        # Calculate final energy demand in TWh from capacities and utilisation
        df["sectoral_consumption"] *= 8760 * df["technology"].map(utilisation) / 1e6
        # Calculate sectoral electricity consumption taking into account technology efficiency
        df["sectoral_consumption"] /= df["technology"].map(efficiency)

        # Aggregate and merge on df_flow
        df = df.drop(columns="technology").groupby(["area", "year"]).sum().reset_index()
        sectoral_consumption = df.loc[
            df["year"] == params["year"], "sectoral_consumption"
        ].sum()
        df_flow = df_flow.merge(df, on=["area", "year"], how="left")

        # (1) Subtract electricity grid losses
        is_electricity_final_use = df_flow["carrier"].str.startswith(
            "electricity"
        ) & df_flow["carrier"].str.endswith("final use")
        df_flow.loc[is_electricity_final_use, "value"] *= 1 - grid_loss
        logging.info(
            f"Subtracting {(grid_loss * 100):.1f}% electricity distribution loss"
        )

        # (2) Subtract sectoral electricity consumption
        df_flow["sectoral_consumption"] *= -1
        is_electricity_LV_final_use = df_flow["carrier"] == "electricity LV final use"
        df_flow.loc[is_electricity_LV_final_use, "value"] = df_flow.loc[
            is_electricity_LV_final_use, ["value", "sectoral_consumption"]
        ].sum(axis=1)
        df_flow = df_flow.drop(columns="sectoral_consumption")
        logging.info(
            f"Subtracting endogenous sectoral electricity consumption: {sectoral_consumption:.1f} TWh ({params['year']})"
        )

        df_flow["value"] = df_flow["value"].round(2)

        return df_flow

    # 1. For 2020, we used to subtract 4 TWh to ensure matching with historical ARE data
    # Now we consider the Eurostat data to be preferred and will modify the ARE-based data in the pre-processing step to ensure the match (not yet done, but irrelevant for long-term runs)

    # 2. Projected electricity final use for 2020-2050 includes neither endogenous electricity consumption nor grid losses
    # No subtraction or addition is needed (grid losses will be determined endogenously, both at the transmission and distribution levels)

    # 3. Reported electricity final use for base year (2024) includes endogenous electricity consumption and all losses
    # We need to subtract endogenous electricity consumption and distribution & transmission losses

    if params["year"] == last_historical_year:
        inputs["annual_energy_flows"] = (
            subtract_endogenous_electricity_consumption_and_losses(
                inputs["annual_energy_flows"], inputs["installed_capacity"]
            )
        )

    # Rescale negative co2 emissions and reduce the variable cost of co2 transport and storage (depending in the year)
    def rescale_cdr(df, cdr_factor=1):
        storage_cost_per_tco2 = 230  # PLN/tCO2
        cdr_techs = ["biogas upgrading", "biomass agriculture CHP CC"]
        is_cdr_tech = df["technology"].isin(cdr_techs)

        df_cdr = df[is_cdr_tech].copy()
        df = df[~is_cdr_tech]

        df_cdr["value"] = df_cdr["value"].astype(float)
        df_cdr = df_cdr.pivot(
            index=["technology_year", "technology"], columns="parameter", values="value"
        ).reset_index()

        df_cdr["variable_cost"] -= (
            df_cdr["co2_emissions"]
            / df_cdr["efficiency"]
            * storage_cost_per_tco2
            * (1 - cdr_factor)
        )
        df_cdr["co2_emissions"] *= cdr_factor
        df_cdr = df_cdr.melt(
            id_vars=["technology_year", "technology"],
            var_name="parameter",
            value_name="value",
        )

        df = pd.concat([df, df_cdr])
        return df

    inputs["technology_cost_data"] = rescale_cdr(
        inputs["technology_cost_data"],
        cdr_factor=cdr_factor[params["year"]],
    )

    # Cleanup the input capacities
    def remove_zero_and_future_capacities(df, investment_technologies, year):
        df = df[(df["nom"] > 0) | df["technology"].isin(investment_technologies)]
        df = df[(df["build_year"] <= year) & (df["retire_year"].fillna(np.inf) >= year)]
        return df

    inputs["installed_capacity"] = remove_zero_and_future_capacities(
        inputs["installed_capacity"],
        params["investment_technologies"],
        params["year"],
    )

    # CALIBRATING INITIAL GRID CAPACITY

    # For 2020 and 2025 the baseline grid is a result of the capacity optimisation with an almost zero unit cost
    # We turn cumulative inf capacities into extendable (non-cumulative) ones
    # Except for vRES connections which are equated with the total vRES capacity

    def make_grid_capacities_extendable(df):
        extendable_grid_technologies = [
            "transformation EHV-HMV",
            "transformation HMV-EHV",
            "distribution HMV",
            "transformation HMV-LV",
            "transformation LV-HMV",
            "distribution LV",
            "connection HMV-vRES",
            "connection vRES-HMV",
            # "direct line vRES",
        ]

        df.loc[
            df["technology"].isin(extendable_grid_technologies)
            & df["technology"].isin(params["investment_technologies"]),
            "cumulative",
        ] = False
        df.loc[
            df["technology"].isin(extendable_grid_technologies)
            & df["technology"].isin(params["investment_technologies"]),
            "nom",
        ] = 0
        df.loc[
            df["technology"].isin(extendable_grid_technologies)
            & df["technology"].isin(params["investment_technologies"]),
            "retire_year",
        ] = np.nan
        return df

    def set_vres_connection_capacity(df):
        df_vres_capacity = (
            df.loc[
                df["technology"].isin(
                    ["solar PV ground", "wind onshore", "wind onshore old"]
                )
                & (df["build_year"] == params["year"])
            ]
            .groupby("area")["nom"]
            .sum()
        )
        is_vres_connection = df["technology"].isin(
            [
                "connection HMV-vRES",
                "connection vRES-HMV",
                # "direct line vRES"
            ]
        )
        df.loc[
            is_vres_connection,
            "nom",
        ] = df.loc[
            is_vres_connection, "area"
        ].map(df_vres_capacity)
        return df

    def rescale_grid_costs(df, factor=1):
        sel = df["technology"].isin(
            [
                "transformation EHV-HMV",
                "distribution HMV",
                "transformation HMV-LV",
                "distribution LV",
                "connection HMV-vRES",
                "transmission line AC",
            ]
        ) & df["parameter"].isin(["fixed_cost", "investment_cost"])
        df.loc[sel, "value"] = (
            pd.to_numeric(df.loc[sel, "value"], errors="coerce") * factor
        )
        return df

    if params["year"] <= 2025:
        inputs["installed_capacity"] = make_grid_capacities_extendable(
            inputs["installed_capacity"]
        )
        inputs["installed_capacity"] = set_vres_connection_capacity(
            inputs["installed_capacity"]
        )
        inputs["technology_cost_data"] = rescale_grid_costs(
            inputs["technology_cost_data"],
            factor=grid_cost_reduction_factor,
        )

    ### SCENARIO DEFINITIONS

    # Set limits to capacity growth of certain technologies
    def set_capacity_constraints(df, carriers_years_values, attr="max_growth"):
        df = df.set_index(["area", "carrier", "year", "attribute"])
        for carrier, year, value in carriers_years_values:
            df.loc[("PL", carrier, year, attr)] = [value]
        df = df.reset_index()
        return df

    # Delayed RES variant: assume 70% of PV addition rate, 50% for wind onshore,
    # and 3-year delay in wind offshore deployment (target capacity in 2040: 12 GW vs. 18 GW in the main scenario)
    if "delayed_RES" in params["run_name"]:
        inputs["capacity_addition_potentials"] = set_capacity_constraints(
            inputs["capacity_addition_potentials"],
            [
                # 2030
                ("wind onshore", 2030, 3750),  # around 13 GW by 2030
                ("wind offshore", 2030, 3340),  # 3-year delay
                ("solar PV roof", 2030, 3500),  # ~ WEM (slightly more)
                ("solar PV ground", 2030, 5250),  # ~ WEM (slightly more)
                # 2035
                ("wind onshore", 2035, 5000),  # around 15 GW by 2035
                ("wind offshore", 2035, 5150),  # 3-year delay
                ("solar PV roof", 2035, 3500),  # ~ WEM (slightly less)
                ("solar PV ground", 2035, 5250),  # ~ WEM (slightly less)
                # 2040
                ("wind onshore", 2040, 5000),  # around 18 GW by 2040
                ("wind offshore", 2040, 3510),  # ~ 12 GW by 2040
                ("solar PV roof", 2040, 3500),  # ~ WEM (slightly less)
                ("solar PV ground", 2040, 5250),  # ~ WEM (slightly less)
            ],
            attr="max_growth",
        )

    # Local PV variant: assume yearly addition rate of prosumer/utility scale PV to be +2 GW/+0.5 GW
    # vs. +1 GW/+1.5 GW in the main scenario (total PV growth is the same in both scenarios, +2.5 GW/year)
    if "local_PV" in params["run_name"]:
        inputs["capacity_addition_potentials"] = set_capacity_constraints(
            inputs["capacity_addition_potentials"],
            [
                # 2030
                ("solar PV roof", 2030, 10000),
                ("solar PV ground", 2030, 2500),
                # 2035
                ("solar PV roof", 2035, 10000),
                ("solar PV ground", 2035, 2500),
                # 2040
                ("solar PV roof", 2040, 10000),
                ("solar PV ground", 2040, 2500),
            ],
            attr="max_growth",
        )

    # Less batteries variant: assume 50% of total battery capacity vs. non-forced optimisation result
    # More batteries variant: assume 150% of total battery capacity vs. non-forced optimisation result
    # Values used from the "instrat_ambitious+trade" run
    battery_attr = None
    if "less_batteries" in params["run_name"]:
        battery_factor = 0.5
        battery_attr = "nom_max"
    if "more_batteries" in params["run_name"]:
        battery_factor = 1.5
        battery_attr = "nom_min"

    if battery_attr is not None:
        inputs["capacity_addition_potentials"] = set_capacity_constraints(
            inputs["capacity_addition_potentials"],
            [
                ("battery large storage", 2030, 23200 * battery_factor),
                ("battery large storage", 2035, 40200 * battery_factor),
                ("battery large storage", 2040, 88300 * battery_factor),
                ("battery large power", 2030, 4420 * battery_factor),
                ("battery large power", 2035, 6940 * battery_factor),
                ("battery large power", 2040, 11440 * battery_factor),
            ],
            attr=battery_attr,
        )

    # Less electrolysis variant: assume low electrolysis capacity
    if "less_electrolysis" in params["run_name"]:
        electrolyser_factor = 0.50
        inputs["capacity_addition_potentials"] = set_capacity_constraints(
            inputs["capacity_addition_potentials"],
            [
                ("hydrogen electrolysis", 2035, 2760 * electrolyser_factor),
                ("hydrogen electrolysis", 2040, 9390 * electrolyser_factor),
            ],
            attr="nom_max",
        )

    # Less heat pumps variant: assume 50% of total heat pump capacity additions after 2030 vs. non-forced optimisation result
    if "less_heat_pumps" in params["run_name"]:
        heat_pump_factor = 0.50
        for attr in ["nom_max", "nom_min"]:
            inputs["capacity_addition_potentials"] = set_capacity_constraints(
                inputs["capacity_addition_potentials"],
                [
                    ("heat pump small", 2030, 11050),
                    ("heat pump small", 2035, 11050 + 14270 * heat_pump_factor),
                    ("heat pump small", 2040, 8090 + (14270 + 9810) * heat_pump_factor),
                ],
                attr=attr,
            )

    # No grid cost variant: set virtually zero grid costs for unlimited grid expansion (after 2025 - in 2025 it is already applied)
    if params["year"] > 2025 and "zero_grid_cost" in params["run_name"]:
        inputs["technology_cost_data"] = rescale_grid_costs(
            inputs["technology_cost_data"],
            factor=grid_cost_reduction_factor,
        )

    # Make sure all capacity utilisation inputs with area "PL" are expanded to all voivodeships when not copperplate
    voivodeships = [
        "PL dolnośląskie",
        "PL kujawsko-pomorskie",
        "PL lubelskie",
        "PL lubuskie",
        "PL łódzkie",
        "PL małopolskie",
        "PL mazowieckie",
        "PL opolskie",
        "PL podkarpackie",
        "PL podlaskie",
        "PL pomorskie",
        "PL śląskie",
        "PL świętokrzyskie",
        "PL warmińsko-mazurskie",
        "PL wielkopolskie",
        "PL zachodniopomorskie",
    ]

    def expand_input_with_voivodeships(df):
        df["area"] = df["area"].apply(
            lambda x: voivodeships + ["PL"] if x == "PL" else [x]
        )
        df = df.explode("area")
        return df

    if not "copperplate" in params["run_name"]:
        inputs["capacity_utilisation"] = expand_input_with_voivodeships(
            inputs["capacity_utilisation"]
        )

    return inputs


def custom_installed_capacity_operation(df, params):

    # Enforce hard coal industrial CHPs to be zero by 2040
    df.loc[
        (df["technology"] == "hard coal CHP")
        & (df["qualifier"] == "industrial")
        & (df["build_year"] >= 2040),
        "nom",
    ] = 0

    # (A) Do not allow capacities of retirement-allowed technologies in years > Y f exceed the optimisation result for year Y
    retirement_technologies = params["retirement_technologies"]

    # (B) Assume a 2.5% annual decline of baseline (2025) grid capacity (old infrastructure, in line with the lifetime of 40 years)
    grid_technologies_old = [
        "transformation EHV-HMV",
        "transformation HMV-EHV",
        "distribution HMV",
        "transformation HMV-LV",
        "transformation LV-HMV",
        "distribution LV",
        # Consider vRES connections as old infrastructure not to underestimate grid expansion costs
        "connection HMV-vRES",
        "connection vRES-HMV",
    ]

    # (B) Assume a decline of baseline (2025) grid capacity in 40 years (new infrastructure, in line with the lifetime of 40 years)
    grid_technologies_new = [
        # "connection HMV-vRES",
        # "connection vRES-HMV",
    ]

    year = params["year"]

    names = df.set_index("name").index
    columns = df.columns

    # Remove year from capacity names
    df["name"] = df["name"].str[: -len(" 2020")]

    # Change nans in area_from and qualifier with a string
    columns_with_nans = [
        "area_from",
        "qualifier",
        "bus_qualifier",
        "bus_from_qualifier",
        "bus2_qualifier",
        "length",
    ]

    for col in columns_with_nans:
        if col in df.columns:
            df[col] = df[col].fillna("none")

    # Drop retire year columns
    df = df.drop(columns=["retire_year"])

    # Change to wide format
    index = [
        "name",
        "area",
        "area_from",
        "technology",
        "qualifier",
        "bus_qualifier",
        "bus_from_qualifier",
        "bus2_qualifier",
        "cumulative",
        "length",
    ]
    df = df.pivot_table(index=index, columns="build_year", values="nom")

    # (A) Cap future capacities with the value at the simulation year
    is_retirement = df.index.get_level_values("technology").isin(
        retirement_technologies
    )
    is_domestic = df.index.get_level_values("area").str.startswith("PL")
    sel = is_retirement & is_domestic
    future_years = df.columns[df.columns > year]

    df.loc[sel, future_years] = np.minimum(
        df.loc[sel, year].values[:, np.newaxis],
        df.loc[sel, future_years].values,
    )

    # (B) Cap future capacities with the 2025 value reduced by 2.5% per year
    # (C) For new grid technologies, assume decline in 40 years
    grid_baseline_year = 2025
    grid_lifetime = 40
    annual_decline = 1.0 / grid_lifetime
    if year == grid_baseline_year:
        is_grid_new = df.index.get_level_values("technology").isin(
            grid_technologies_new
        )
        is_grid_old = df.index.get_level_values("technology").isin(
            grid_technologies_old
        )
        is_domestic = df.index.get_level_values("area").str.startswith("PL")
        future_years = df.columns[df.columns > grid_baseline_year]
        sel = is_grid_old & is_domestic
        for y in future_years:
            df.loc[sel, y] = np.maximum(
                df.loc[sel, grid_baseline_year]
                * (1 - (y - grid_baseline_year) * annual_decline),
                0,
            ).round(0)
        sel = is_grid_new & is_domestic
        for y in future_years:
            df.loc[sel, y] = (
                df.loc[sel, grid_baseline_year]
                if y - grid_baseline_year < grid_lifetime
                else 0
            )

    # Change back to long format
    df = df.reset_index()
    df = df.melt(id_vars=index, var_name="build_year", value_name="nom")
    df["name"] = df["name"] + " " + df["build_year"].astype(str)
    df["retire_year"] = np.where(df["cumulative"] == "TRUE", df["build_year"], np.nan)
    for col in columns_with_nans:
        if col in df.columns:
            df[col] = df[col].replace("none", np.nan)

    df = df[columns]
    df = df.set_index("name").loc[names].reset_index()

    return df


def run_model(
    param_dict,
    year_start=2025,
    year_end=2040,
    load_network_only=False,
    export_output_files=True,
):

    for i, (scenario, year_params) in enumerate(param_dict.items()):
        for j, (year, params) in enumerate(year_params.items()):
            if year < year_start or year > year_end:
                continue
            logging.info("*** RUN: " + params["run_name"] + " " + "*" * 30)
            if not load_network_only:
                network = run_simulation(
                    params=params,
                    custom_input_operation=custom_input_operation,
                    installed_capacity_variant=(
                        scenario if year >= 2025 else None
                    ),  # do not save optimisation results before 2025
                    custom_installed_capacity_operation=custom_installed_capacity_operation,
                )
            else:
                network = load_network(params["run_name"])

            if export_output_files:
                logging.info("Generating outputs of run: " + params["run_name"])
                output_data_dir = lambda *path: data_dir(
                    "clean", "outputs", scenario, str(year), *path
                )
                generate_outputs(
                    network, output_plots_dir=None, output_data_dir=output_data_dir
                )

            if (i < len(param_dict) - 1) and (j < len(year_params) - 1):
                # Free memory before next scenario (unless last one)
                del network
                gc.collect()

    return network


if __name__ == "__main__":

    param_dict = define_scenarios(delete_capacity_file=False)
    number_of_scenarios = len(param_dict)

    # Get from command line number of param dict element to simulate
    param_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Get from command line year to simulate
    year_to_simulate = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

    # Keep only selected param dict element
    param_key = list(param_dict.keys())[param_index]
    param_dict = {param_key: param_dict[param_key]}
    print(
        f"Simulating scenario {param_index+1}/{number_of_scenarios} ({param_key}) for year {year_to_simulate}"
    )

    network = run_model(
        param_dict,
        year_start=year_to_simulate,
        year_end=year_to_simulate,
        load_network_only=False,
    )
