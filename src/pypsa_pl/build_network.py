import pandas as pd
import numpy as np
import pypsa
import logging

from pypsa_pl.helper_functions import ignore_user_warnings, ignore_future_warnings
from pypsa_pl.config import data_dir
from pypsa_pl.custom_line_types import add_custom_line_types
from pypsa_pl.make_time_profiles import set_snapshot_index, make_profile_funcs
from pypsa_pl.mathematical_operations import calculate_annuity


def read_input(name, variant):
    try:
        df = pd.read_csv(
            data_dir(
                "input",
                f"{name};variant={variant}.csv",
            )
        )
    except:
        logging.error(f"{name};variant={variant}.csv not found!")
        df = None
    return df


def concat_inputs(name, variants_list):
    if not isinstance(variants_list, list):
        variants_list = [variants_list]

    if len(variants_list) == 0:
        return None

    df = pd.concat(
        [read_input(name, variant) for variant in variants_list], ignore_index=True
    )
    return df


def load_and_preprocess_inputs(params, custom_operation=None):

    inputs = {
        name: concat_inputs(name, params[name])
        for name in [
            "technology_carrier_definitions",
            "technology_cost_data",
            "installed_capacity",
            "annual_energy_flows",
            "capacity_utilisation",
            "capacity_addition_potentials",
        ]
    }

    if custom_operation is not None:
        inputs = custom_operation(inputs, params)

    # Pivot df_tech to wide format
    df_tech = inputs["technology_cost_data"].pivot(
        index=["technology", "technology_year"],
        columns="parameter",
        values="value",
    )
    # Convert values to floats if possible
    df_tech = df_tech.apply(pd.to_numeric, errors="ignore").reset_index()

    # Get co2_cost from df_tech
    df_co2_cost = df_tech[["technology_year", "co2_cost"]].dropna()
    # Drop co2 emissions technology from df_tech
    df_tech = df_tech[df_tech["technology"] != "co2 emissions"].drop(columns="co2_cost")

    inputs["technology_cost_data"] = df_tech
    inputs["co2_cost"] = df_co2_cost

    # In capacity data, replace the "inf" value with a large number, virtually infinite in the modelling context
    inputs["installed_capacity"]["nom"] = (
        inputs["installed_capacity"]["nom"].replace(np.inf, params["inf"]).astype(float)
    )

    # Extract df_final use from df_flow to get p_set_annual atribute
    df_flow = inputs["annual_energy_flows"]
    is_final_use = df_flow["carrier"].str.contains("final use")
    df_final_use = df_flow[is_final_use].copy()
    df_final_use = df_final_use[
        (df_final_use["parameter"] == "flow") & (df_final_use["type"] == "inflow")
    ]

    # Pivot df_util and df_final_use to wide format

    df_util = (
        inputs["capacity_utilisation"]
        .pivot(
            index=["area", "technology", "qualifier", "year"],
            columns="parameter",
            values="value",
        )
        .reset_index()
    )
    df_final_use = df_final_use.pivot(
        index=["area", "carrier", "year"],
        columns="parameter",
        values="value",
    ).reset_index()

    # In df_final_use convert flow (in TWh) to p_set_annual (in MW)
    p_set_annual = df_final_use["flow"] * 1e6 / 8760
    is_constant = ~df_final_use["carrier"].isin(
        [
            "electricity final use",
            "electricity EHV final use",
            "electricity HMV final use",
            "electricity LV final use",
            "hydrogen final use",
            "light vehicle mobility final use",
            "space heating final use",
            "water heating final use",
            "other heating final use",
        ]
    )
    df_final_use.loc[is_constant, "p_set"] = p_set_annual[is_constant]
    df_final_use.loc[~is_constant, "p_set_annual"] = p_set_annual[~is_constant]
    df_final_use = df_final_use.drop(columns="flow")

    inputs["capacity_utilisation"] = df_util
    inputs["final_use"] = df_final_use

    return inputs


def create_custom_network(params):
    # Get default components
    components = pypsa.components.components.copy()

    # Define custom components
    # columns ["list_name", "description", "type"]
    components.loc["Area"] = ["areas", "geographical location", np.nan]

    # Get default component attributes
    attrs = pypsa.definitions.structures.Dict(
        {k: v.copy() for k, v in pypsa.components.component_attrs.items()}
    )
    # Define custom atributes
    # columns: ["type", "unit", "default", "description", "status"]
    for component in ["Area"]:
        attrs[component] = pd.DataFrame(
            columns=["type", "unit", "default", "description", "status"]
        ).rename_axis("attribute")
        attrs[component].loc["name"] = [
            "string",
            np.nan,
            np.nan,
            "name of the area",
            "Input (required)",
        ]
        attrs[component].loc["x"] = [
            "float",
            np.nan,
            0,
            "Position (e.g. longitude)",
            "Input (optional)",
        ]
        attrs[component].loc["y"] = [
            "float",
            np.nan,
            0,
            "Position (e.g. latitude)",
            "Input (optional)",
        ]
    for component in ["GlobalConstraint"]:
        attrs[component].loc["area"] = [
            "string",
            np.nan,
            np.nan,
            "geographical location to which constraint is applied",
            "Input (required)",
        ]
    for component in ["Bus", "Generator", "Link", "Line", "Store"]:
        attrs[component].loc["area"] = [
            "string",
            np.nan,
            np.nan,
            "geographical location of an asset or a bus (output area for lines and links connecting two different areas)",
            "Input (required)",
        ]
        if component in ["Link", "Line"]:
            attrs[component].loc["area_from"] = [
                "string",
                np.nan,
                np.nan,
                "input area of a line or a link if it connects two different areas",
                "Input (optional)",
            ]
        attrs[component].loc["qualifier"] = [
            "string",
            np.nan,
            np.nan,
            "extra details about an asset or a bus influencing its modelled behaviour",
            "Input (optional)",
        ]
    for component in ["Generator", "Link", "Line", "Store"]:
        attrs[component].loc["technology"] = [
            "string",
            np.nan,
            np.nan,
            "specific technology of an asset",
            "Input (required)",
        ]
        attrs[component].loc["aggregation"] = [
            "string",
            np.nan,
            np.nan,
            "aggregation category",
            "Input (optional)",
        ]
        if component == "Link":
            attrs[component].loc["length"] = [
                "float",
                "km",
                np.nan,
                "length of the link in km (optional, used for DC lines)",
                "Input (optional)",
            ]
        if component != "Line":
            attrs[component].loc["variable_cost"] = [
                "static or series",
                "currency/MWh",
                0,
                "variable cost of production, excluding CO2 cost",
                "Input (optional)",
            ]
            attrs[component].loc["co2_cost"] = [
                "static or series",
                "currency/MWh",
                0,
                "CO2 cost component of variable cost of production",
                "Input (optional)",
            ]
        attrs[component].loc["fixed_cost"] = [
            "float",
            "currency/MW",
            0,
            "fixed annual O&M cost of maintaining 1 MW (or 1 MWkm) of capacity",
            "Input (optional)",
        ]
        attrs[component].loc["investment_cost"] = [
            "float",
            "currency/MW",
            0,
            "total overnight investment cost of extending capacity by 1 MW (or 1 MWkm)",
            "Input (optional)",
        ]
        attrs[component].loc["annual_investment_cost"] = [
            "float",
            "currency/MW",
            0,
            "annualised investment cost of extending capacity by 1 MW (or 1 MWkm)",
            "Input (optional)",
        ]
        attrs[component].loc["parent"] = [
            "string",
            np.nan,
            np.nan,
            "parent capacity to which the asset is related",
            "Input (optional)",
        ]
        attrs[component].loc["parent_ratio"] = [
            "float",
            np.nan,
            np.nan,
            "ratio of the asset's capacity to the parent capacity",
            "Input (optional)",
        ]
    for component in ["Generator", "Link"]:
        attrs[component].loc["p_min_pu_annual"] = [
            "float",
            np.nan,
            0,
            "minimum annual capacity utilisation factor",
            "Input (optional)",
        ]
        attrs[component].loc["p_max_pu_annual"] = [
            "float",
            np.nan,
            1,
            "maximum annual capacity utilisation factor",
            "Input (optional)",
        ]
        attrs[component].loc["p_set_pu_annual"] = [
            "float",
            np.nan,
            np.nan,
            "annual capacity utilisation factor (enforced if not NaN)",
            "Input (optional)",
        ]
    for component in ["Carrier"]:
        attrs[component].loc["order"] = [
            "int",
            np.nan,
            0,
            "rank of carrier used for ordering in plots",
            "Input (optional)",
        ]
        attrs[component].loc["aggregation"] = [
            "string",
            np.nan,
            np.nan,
            "aggregation category of carrier",
            "Input (optional)",
        ]

    # Create network with custom attributes
    network = pypsa.Network(
        override_components=components, override_component_attrs=attrs
    )
    network.name = params["run_name"]
    network.meta = params
    # Add custom line types
    add_custom_line_types(network)

    return network


@ignore_user_warnings
def load_network(run_name):
    network = create_custom_network(params={"run_name": run_name})
    network.import_from_csv_folder(data_dir("runs", run_name, "output_network"))
    return network


def add_snapshots(network, params):
    # Read electricity demand profile to get snapshots
    df_t = pd.read_csv(
        data_dir(
            "input",
            f"timeseries;variant={params['timeseries']}",
            f"demand_profile;carrier=electricity final use;year={params['weather_year']}.csv",
        ),
        usecols=["hour"],
    )

    df_t = set_snapshot_index(df_t, params)

    network.set_snapshots(df_t.index)

    # We have to increase snapshot weightings to ensure normalization to a year
    network.snapshot_weightings.loc[:, "objective"] = 8760 / len(network.snapshots)
    network.snapshot_weightings.loc[:, "generators"] = 8760 / len(network.snapshots)
    # Weighting cannot be set for stores as it leads to wrong results
    network.snapshot_weightings.loc[:, "stores"] = 1


def add_carriers(network, inputs, params):

    # Merge df_carr with co2_emissions column of df_tech
    # IMPORTANT: co2_emissions within the carrier need to be independent of the specific technology
    df_carr = inputs["technology_carrier_definitions"]
    df_tech = inputs["technology_cost_data"]

    df = df_carr[["carrier", "color", "order", "aggregation", "technology"]].merge(
        df_tech.loc[
            df_tech["technology_year"] == params["year"],
            ["technology", "co2_emissions"],
        ],
        on="technology",
        how="left",
    )

    df = df.drop(columns="technology").groupby("carrier").first()

    network.remove("Carrier", network.carriers.index)
    network.add("Carrier", df.index, **df)


def determine_bus_qualifiers(df):
    # Capacities with output carrier
    has_output_carrier = df["output_carrier"].notna()
    df.loc[has_output_carrier, "output_qualifier"] = df.loc[
        has_output_carrier, "bus_qualifier"
    ]
    # Capacities with input carrier
    has_input_carrier = df["input_carrier"].notna()
    df.loc[has_input_carrier, "input_qualifier"] = df.loc[
        has_input_carrier, "bus_from_qualifier"
    ]
    # Capacities with output2 carrier
    has_output_carrier = df["output2_carrier"].notna()
    df.loc[has_output_carrier, "output2_qualifier"] = df.loc[
        has_output_carrier, "bus2_qualifier"
    ]
    # Only capacities with bus carrier can keep bus qualifier at the end
    has_bus_carrier = df["bus_carrier"].notna()
    df.loc[~has_bus_carrier, "bus_qualifier"] = np.nan
    return df


def add_buses_and_areas(network, inputs, params):

    df_cap = inputs["installed_capacity"]
    df_carr = inputs["technology_carrier_definitions"]
    # List all technology, area, and bus qualifier combinations present in the capacities
    df = df_cap[
        [
            "technology",
            "area",
            "area_from",
            "bus_qualifier",
            "bus_from_qualifier",
            "bus2_qualifier",
        ]
    ].drop_duplicates()

    # Combine with bus carriers of techs input and output buses
    bus_carrier_columns = [
        "bus_carrier",
        "input_carrier",
        "output_carrier",
        "output2_carrier",
    ]
    df = df.merge(
        df_carr[["technology", *bus_carrier_columns]], on="technology", how="inner"
    )

    # Determine bus qualifiers
    df = determine_bus_qualifiers(df)

    # Identify all unique area, bus carrier, and bus qualifier combinations
    # Destination buses (area, bus, bus2)
    df_to = pd.concat(
        [
            df[["area", col, col.replace("carrier", "qualifier")]]
            .rename(columns=lambda x: "area" if x == "area" else x.split("_")[1])
            .dropna(subset=["carrier"])
            for col in ["bus_carrier", "output_carrier", "output2_carrier"]
        ]
    )

    # Origin buses (area_from, bus_from)
    df["area_from"] = df["area_from"].fillna(df["area"])
    df_from = pd.concat(
        [
            df[["area_from", col, col.replace("carrier", "qualifier")]]
            .rename(columns=lambda x: "area" if x == "area_from" else x.split("_")[1])
            .dropna(subset=["carrier"])
            for col in ["bus_carrier", "input_carrier"]
        ]
    )

    df = pd.concat([df_to, df_from]).drop_duplicates()

    # Define bus names
    df["bus"] = df["area"] + " " + df["carrier"]
    has_qualifier = df["qualifier"].notna()
    df.loc[has_qualifier, "bus"] += " " + df.loc[has_qualifier, "qualifier"]
    df = df.sort_values("bus")

    # Define voltage level for EHV buses (400 kV)
    df["v_nom"] = 1.0  # default value
    df.loc[df["qualifier"] == "EHV", "v_nom"] = 400.0

    network.remove("Bus", network.buses.index)
    network.add("Bus", df.set_index("bus").index, **df.set_index("bus"))

    # Unique areas
    df = df[["area"]].copy()
    # Always add "PL" area
    df = pd.concat([df, pd.DataFrame([["PL"]], columns=["area"])], ignore_index=True)
    df = df.drop_duplicates().sort_values("area")

    network.remove("Area", network.areas.index)
    network.add("Area", df.set_index("area").index, **df.set_index("area"))


def process_capacity_data(inputs, params):

    df_cap = inputs["installed_capacity"]
    df_carr = inputs["technology_carrier_definitions"]
    df_tech = inputs["technology_cost_data"]
    df_co2_cost = inputs["co2_cost"]
    df_util = inputs["capacity_utilisation"]
    df_final_use = inputs["final_use"]

    # (I) Determine carrier, buses, sign, and p_nom/e_nom

    df = df_cap.merge(
        df_carr[
            [
                "technology",
                "carrier",
                "aggregation",
                "bus_carrier",
                "input_carrier",
                "output_carrier",
                "output2_carrier",
                "component",
                "parent",
            ]
        ],
        on="technology",
        how="inner",
    )

    df = determine_bus_qualifiers(df)
    df = df.drop(columns=["bus_from_qualifier", "bus2_qualifier"])

    # (1) Generators
    is_gen = df["component"] == "Generator"
    is_positive_gen = is_gen & df["input_carrier"].isna()
    is_negative_gen = is_gen & df["output_carrier"].isna()

    df.loc[is_positive_gen, "bus"] = (
        df.loc[is_positive_gen, "area"]
        + " "
        + df.loc[is_positive_gen, "output_carrier"]
    )
    has_qualifier = is_positive_gen & df["output_qualifier"].notna()
    df.loc[has_qualifier, "bus"] += " " + df.loc[has_qualifier, "output_qualifier"]

    df.loc[is_negative_gen, "bus"] = (
        df.loc[is_negative_gen, "area"] + " " + df.loc[is_negative_gen, "input_carrier"]
    )
    has_qualifier = is_negative_gen & df["input_qualifier"].notna()
    df.loc[has_qualifier, "bus"] += " " + df.loc[has_qualifier, "input_qualifier"]

    df.loc[is_positive_gen, "sign"] = 1
    df.loc[is_negative_gen, "sign"] = -1
    df.loc[is_gen, "p_nom"] = df.loc[is_gen, "nom"]

    # (2) Links and lines
    is_link = df["component"] == "Link"
    is_line = df["component"] == "Line"
    is_link_or_line = is_link | is_line
    if "area_from" not in df.columns:
        df.loc[is_link_or_line, "area_from"] = df.loc[is_link_or_line, "area"]
    else:
        df.loc[is_link_or_line, "area_from"] = df.loc[
            is_link_or_line, "area_from"
        ].fillna(df.loc[is_link_or_line, "area"])

    df.loc[is_link_or_line, "bus_input"] = (
        df.loc[is_link_or_line, "area_from"]
        + " "
        + df.loc[is_link_or_line, "input_carrier"]
    )
    has_qualifier = is_link_or_line & df["input_qualifier"].notna()
    df.loc[has_qualifier, "bus_input"] += " " + df.loc[has_qualifier, "input_qualifier"]

    df.loc[is_link_or_line, "bus_output"] = (
        df.loc[is_link_or_line, "area"]
        + " "
        + df.loc[is_link_or_line, "output_carrier"]
    )
    has_qualifier = is_link_or_line & df["output_qualifier"].notna()
    df.loc[has_qualifier, "bus_output"] += (
        " " + df.loc[has_qualifier, "output_qualifier"]
    )

    has_output2 = is_link_or_line & df["output2_carrier"].notna()
    df.loc[has_output2, "bus_output2"] = (
        df.loc[has_output2, "area"] + " " + df.loc[has_output2, "output2_carrier"]
    )
    has_qualifier = has_output2 & df["output2_qualifier"].notna()
    df.loc[has_qualifier, "bus_output2"] += (
        " " + df.loc[has_qualifier, "output2_qualifier"]
    )
    if "bus_output2" not in df.columns:
        df["bus_output2"] = np.nan

    df.loc[is_link, "p_nom"] = df.loc[is_link, "nom"]
    df.loc[is_line, "s_nom"] = df.loc[is_line, "nom"]

    # (3) Stores
    is_store = df["component"] == "Store"

    df.loc[is_store, "bus"] = (
        df.loc[is_store, "area"] + " " + df.loc[is_store, "bus_carrier"]
    )
    has_qualifier = is_store & df["bus_qualifier"].notna()
    df.loc[has_qualifier, "bus"] += " " + df.loc[has_qualifier, "bus_qualifier"]

    df.loc[is_store, "e_nom"] = df.loc[is_store, "nom"]

    df = df.drop(
        columns=[
            "bus_carrier",
            "input_carrier",
            "output_carrier",
            "output2_carrier",
            "bus_qualifier",
            "input_qualifier",
            "output_qualifier",
            "output2_qualifier",
        ]
    )

    # (II) Determine technological and cost parameters

    # Determine technology year
    # (1) If capacity is cumulative and its (virtual) build year can be found in df_tech, use it as technology year
    df_years = df_tech[["technology", "technology_year"]].drop_duplicates()
    df_years["build_year"] = df_years["technology_year"]
    df = df.merge(df_years, on=["technology", "build_year"], how="left")
    df.loc[~df["cumulative"], "technology_year"] = np.nan

    # (2) If not, use the formula: technology_year = 5 * ceil(build_year / 5) - 5
    # e.g. build_year=2020 -> technology_year=2015, build_year=2021 -> technology_year=2020
    df["technology_year"] = (
        df["technology_year"]
        .fillna(5 * (np.ceil(df["build_year"] / 5) - 1))
        .astype(int)
    )

    df = df.merge(df_tech, on=["technology", "technology_year"], how="left")

    # CO2 cost is always from the year of the simulation
    co2_cost = df_co2_cost.loc[
        df_co2_cost["technology_year"] == params["year"], "co2_cost"
    ].iloc[0]
    df["co2_cost"] = co2_cost
    # df = df.merge(df_co2_cost, on="technology_year", how="left")

    # Set default efficiency = 1
    df["efficiency"] = (
        1.0 if "efficiency" not in df.columns else df["efficiency"].fillna(1.0)
    )

    # Stores - set default standing losses = 0
    df["standing_loss"] = (
        0 if "standing_loss" not in df.columns else df["standing_loss"].fillna(0)
    )

    # Lines and links - make sure length column exists
    if "length" not in df.columns:
        df["length"] = np.nan

    # Lines - make sure line_type column exists
    if "line_type" not in df.columns:
        df["line_type"] = np.nan

    if "efficiency2" not in df.columns:
        df["efficiency2"] = np.nan

    # Calculate marginal cost
    df["variable_cost"] = df["variable_cost"].fillna(0)
    df["co2_cost"] = df["co2_emissions"].fillna(0) / df["efficiency"] * df["co2_cost"]
    df["marginal_cost"] = df["variable_cost"] + df["co2_cost"]

    # If not provided, determine retire year by technological lifetime
    df["retire_year"] = df["retire_year"].fillna(df["build_year"] + df["lifetime"] - 1)
    # Then calculate the actual lifetime
    df["lifetime"] = df["retire_year"] - df["build_year"] + 1
    # Select capacities based on build and retire year
    df = df[
        (df["build_year"] <= params["year"]) & (df["retire_year"] >= params["year"])
    ]

    # Calculate capital cost
    df["fixed_cost"] = (
        0 if "fixed_cost" not in df.columns else df["fixed_cost"].fillna(0)
    )
    df["investment_cost"] = (
        0 if "investment_cost" not in df.columns else df["investment_cost"].fillna(0)
    )
    # Consider investment costs only for capacities not specified as cumulative and with build year larger than investment cost start year
    investment_cost_start_year = params.get("investment_cost_start_year", 0)
    has_investment_cost = ~df["cumulative"] & (
        df["build_year"] >= investment_cost_start_year
    )
    df.loc[~has_investment_cost, "investment_cost"] = 0
    df.loc[~has_investment_cost, "annual_investment_cost"] = 0
    df.loc[has_investment_cost, "annual_investment_cost"] = df.loc[
        has_investment_cost, "investment_cost"
    ] * calculate_annuity(
        lifetime=df.loc[has_investment_cost, "lifetime"],
        discount_rate=params["discount_rate"],
    )

    df["capital_cost"] = df["fixed_cost"] + df["annual_investment_cost"]

    # Keep non-zero investment cost only for capacities that are built in the simulation year
    df.loc[df["build_year"] != params["year"], "investment_cost"] = 0

    # (III) Determine extendability
    is_domestic = df["area"].str.startswith("PL")
    is_active = df["build_year"] == params["year"]
    is_cumulative = df["cumulative"]
    is_to_invest = (
        df["technology"].isin(params["investment_technologies"])
        & is_domestic
        & is_active
        & ~is_cumulative
    )
    is_to_retire = (
        df["technology"].isin(params["retirement_technologies"])
        & is_domestic
        & is_active
        & is_cumulative
    )
    if not params.get("optimise_industrial_capacities", False):
        is_industrial = df["qualifier"].fillna("") == "industrial"
        is_to_invest &= ~is_industrial
        is_to_retire &= ~is_industrial

    is_gen_or_link = is_gen | is_link_or_line
    for is_component, nom in [
        (is_gen_or_link, "p_nom"),
        (is_line, "s_nom"),
        (is_store, "e_nom"),
    ]:
        if params.get("invest_from_zero", True):
            df.loc[is_component & is_to_invest, nom] = 0

        # By default set no extendability and nom = nom_min = nom_max
        df.loc[is_component, f"{nom}_extendable"] = False
        df.loc[is_component, f"{nom}_min"] = df.loc[is_component, nom]
        df.loc[is_component, f"{nom}_max"] = df.loc[is_component, nom]
        # Allow extendability for investment and retirement capacities
        df.loc[is_component & (is_to_invest | is_to_retire), f"{nom}_extendable"] = True
        # For investment-allowed capacities: nom < nom_opt < inf
        df.loc[is_component & is_to_invest, f"{nom}_max"] = np.inf
        # For retirement-allowed capacities: 0 < nom_opt < nom
        df.loc[is_component & is_to_retire, f"{nom}_min"] = 0

    # (IV) Incorporate final use assumptions
    # df_final_use might contain the following attributes: p_set_annual, p_set

    df_final_use = df_final_use.rename(columns={"year": "build_year"})
    df = df.merge(df_final_use, on=["area", "carrier", "build_year"], how="left")

    # (V) Incorporate capacity utilisation assumptions
    # df_util might contain the following attributes:
    # p_min_pu, p_max_pu, p_set_pu, p_set_pu_annual
    # e_min_pu, e_max_pu, s_max_pu

    df["qualifier"] = df["qualifier"].fillna("none")
    df_util["qualifier"] = df_util["qualifier"].fillna("none")
    df_util = df_util.rename(columns={"year": "build_year"})

    df = df.merge(
        df_util, on=["area", "technology", "qualifier", "build_year"], how="left"
    )
    df["qualifier"] = df["qualifier"].replace("none", np.nan)

    df["p_min_pu"] = 0.0 if "p_min_pu" not in df.columns else df["p_min_pu"].fillna(0.0)
    df["p_max_pu"] = 1.0 if "p_max_pu" not in df.columns else df["p_max_pu"].fillna(1.0)
    df["e_min_pu"] = 0.0 if "e_min_pu" not in df.columns else df["e_min_pu"].fillna(0.0)
    df["e_max_pu"] = 1.0 if "e_max_pu" not in df.columns else df["e_max_pu"].fillna(1.0)
    df["s_max_pu"] = 1.0 if "s_max_pu" not in df.columns else df["s_max_pu"].fillna(1.0)

    if "p_set_pu" in df.columns:
        p_set = df["p_set_pu"] * df["p_nom"]
        df["p_set"] = df["p_set"].fillna(p_set) if "p_set" in df.columns else p_set
        df = df.drop(columns="p_set_pu")

    if "p_set_pu_annual" in df.columns:
        p_set_annual = df["p_set_pu_annual"] * df["p_nom"]
        df["p_set_annual"] = (
            df["p_set_annual"].fillna(p_set_annual)
            if "p_set_annual" in df.columns
            else p_set_annual
        )
    else:
        df["p_set_pu_annual"] = np.nan

    # df_tech might have defined p_min_pu_annual and p_max_pu_annual for specific technologies
    df["p_min_pu_annual"] = (
        0.0
        if "p_min_pu_annual" not in df.columns
        else df["p_min_pu_annual"].fillna(0.0)
    )
    df["p_max_pu_annual"] = (
        1.0
        if "p_max_pu_annual" not in df.columns
        else df["p_max_pu_annual"].fillna(1.0)
    )
    # Also availability_correction_factor might be defined for vRES technologies
    df["availability_correction_factor"] = (
        1.0
        if "availability_correction_factor" not in df.columns
        else df["availability_correction_factor"].fillna(1.0)
    )

    # Attribute p_set_pu and p_set_pu_annual to heating, mobility, and hydrogen links
    # p_set applies to links that have a constant output
    # p_set_annual applies to links that have a time-dependent output
    for technology, attr in [
        ("centralised space heating", "p_set_pu_annual"),
        ("centralised water heating", "p_set_pu"),
        ("centralised other heating", "p_set_pu"),
        ("decentralised space heating", "p_set_pu_annual"),
        ("decentralised water heating", "p_set_pu"),
        ("building retrofits", "p_set_pu_annual"),
        ("light vehicle mobility", "p_set_pu_annual"),
        ("hydrogen", "p_set_pu"),
    ]:
        if technology == "building retrofits":
            attr_value = params["heat_capacity_utilisation"]
        else:
            attr_value = params[
                technology.replace("decentralised ", "")
                .replace("centralised ", "")
                .replace(" ", "_")
                + "_utilisation"
            ]
        df.loc[df["technology"] == technology, attr] = attr_value

    # (VI) Map parent capacities to children capacities for technologies that have parents
    has_parent_tech = df["parent"].notna()
    df_children = df.loc[
        has_parent_tech,
        [
            "name",
            "area",
            "build_year",
            "technology",
            "qualifier",
            "parent",
        ],
    ]
    parent_techs = df_children["parent"].unique()
    df_parents = df.loc[
        df["technology"].isin(parent_techs),
        ["name", "area", "build_year", "technology", "qualifier"],
    ].rename(columns=lambda x: "parent_" + x)
    df_map = pd.merge(
        df_children,
        df_parents,
        left_on=["parent", "area", "build_year"],
        right_on=["parent_technology", "parent_area", "parent_build_year"],
        how="inner",
    )
    # TODO: solve the issue of non-unique matches in a more general way
    # Each child can have only one parent and each parent can have only one child per technology
    # For non-unique matches, demand that qualifiers match as well
    get_non_unique = lambda x: x.duplicated(subset=["name"], keep=False) | x.duplicated(
        subset=["technology", "parent_name"], keep=False
    )
    is_unique = ~get_non_unique(df_map)
    qualifiers_equal = df_map["qualifier"].fillna("") == df_map[
        "parent_qualifier"
    ].fillna("")
    df_map = df_map[is_unique | qualifiers_equal]
    # If still there is no unique match, demand that the 1 last word name match, 2 last words match, 3 last words match (independent of order)
    for n in range(1, 3 + 1):
        is_unique = ~get_non_unique(df_map)
        last_n_words_equal = df_map["name"].str.split().apply(
            lambda x: set(x[-n:])
        ) == df_map["parent_name"].str.split().apply(lambda x: set(x[-n:]))
        df_map = df_map[is_unique | last_n_words_equal]
    assert sum(get_non_unique(df_map)) == 0

    # Map parent capacities to children capacities
    df_map = df_map[["name", "parent_name"]].rename(columns={"parent_name": "parent"})
    df = df.drop(columns="parent").merge(df_map, on="name", how="left")

    # df = df.dropna(axis=1, how="all")
    df = df.sort_values("name")

    return df


@ignore_future_warnings
def add_capacities(network, df_cap, df_attr_t, params):

    for component, df in df_cap.groupby("component"):

        network.remove(
            component, network.static(component).index.intersection(df["name"])
        )

        index_t = ["carrier", "technology", "qualifier"]

        df_t = df.merge(df_attr_t, on=index_t, how="inner")

        df_attrs_t = (
            df_t.groupby(index_t)
            .agg({"attribute": lambda x: ",".join(sorted(x.unique()))})
            .rename(columns={"attribute": "attrs_t"})
            .reset_index()
        )

        df = df.merge(df_attrs_t, on=index_t, how="left")
        df["attrs_t"] = df["attrs_t"].fillna("")

        for attrs_t, df in df.groupby("attrs_t"):
            attrs_t = attrs_t.split(",")

            df_t = df.merge(df_attr_t, on=index_t, how="inner")
            dfs_t = {
                attr: pd.concat(
                    [
                        make_profile_funcs[profile_type](
                            df_t, network.snapshots, params
                        )
                        for profile_type, df_t in df_t.groupby("profile_type")
                    ],
                    axis=1,
                )
                .dropna(axis=1, how="all")
                .sort_index(axis=1)
                for attr, df_t in df_t.groupby("attribute")
            }

            df.loc[df["qualifier"] == "none", "qualifier"] = np.nan
            df = df.set_index("name")

            if "p_set" in dfs_t:
                # TODO: find out why this fix for p_set is necessary
                dfs_t["p_set"] = dfs_t["p_set"].reindex(
                    df.index, axis=1, fill_value=np.nan
                )

            if component == "Generator":

                network.add(
                    component,
                    df.index,
                    bus=df["bus"],
                    area=df["area"],
                    carrier=df["carrier"],
                    technology=df["technology"],
                    qualifier=df["qualifier"],
                    aggregation=df["aggregation"],
                    p_nom=df["p_nom"],
                    p_nom_extendable=df["p_nom_extendable"],
                    p_nom_min=df["p_nom_min"],
                    p_nom_max=df["p_nom_max"],
                    sign=df["sign"],
                    efficiency=df["efficiency"],
                    variable_cost=df["variable_cost"],
                    co2_cost=df["co2_cost"],
                    marginal_cost=df["marginal_cost"],
                    fixed_cost=df["fixed_cost"],
                    investment_cost=df["investment_cost"],
                    annual_investment_cost=df["annual_investment_cost"],
                    capital_cost=df["capital_cost"],
                    p_min_pu=(dfs_t if "p_min_pu" in attrs_t else df)["p_min_pu"],
                    p_max_pu=(dfs_t if "p_max_pu" in attrs_t else df)["p_max_pu"],
                    p_set=dfs_t["p_set"] if "p_set" in attrs_t else np.nan,
                    p_min_pu_annual=df["p_min_pu_annual"],
                    p_max_pu_annual=df["p_max_pu_annual"],
                    p_set_pu_annual=df["p_set_pu_annual"],
                    build_year=df["build_year"],
                    lifetime=df["lifetime"],
                    parent=df["parent"],
                    parent_ratio=df["parent_ratio"],
                )
                network.generators = network.static(component).sort_index()

            elif component == "Link":

                if not params["reverse_links"]:
                    network.add(
                        component,
                        df.index,
                        bus0=df["bus_input"],
                        bus1=df["bus_output"],
                        bus2=(
                            df["bus_output2"].fillna("") if "bus_output2" in df else ""
                        ),
                        area=df["area"],
                        area_from=df["area_from"],
                        carrier=df["carrier"],
                        technology=df["technology"],
                        qualifier=df["qualifier"],
                        aggregation=df["aggregation"],
                        p_nom=df["p_nom"] / df["efficiency"],
                        p_nom_extendable=df["p_nom_extendable"],
                        p_nom_min=df["p_nom_min"] / df["efficiency"],
                        p_nom_max=df["p_nom_max"] / df["efficiency"],
                        efficiency=(dfs_t if "efficiency" in attrs_t else df)[
                            "efficiency"
                        ],
                        efficiency2=(
                            df["efficiency2"] if "efficiency2" in df else np.nan
                        ),
                        length=df["length"],
                        variable_cost=df["variable_cost"] * df["efficiency"],
                        co2_cost=df["co2_cost"],
                        marginal_cost=df["marginal_cost"] * df["efficiency"],
                        fixed_cost=df["fixed_cost"]
                        * df["length"].fillna(1.0)
                        * df["efficiency"],
                        investment_cost=df["investment_cost"]
                        * df["length"].fillna(1.0)
                        * df["efficiency"],
                        annual_investment_cost=df["annual_investment_cost"]
                        * df["length"].fillna(1.0)
                        * df["efficiency"],
                        capital_cost=df["capital_cost"]
                        * df["length"].fillna(1.0)
                        * df["efficiency"],
                        p_min_pu=(dfs_t if "p_min_pu" in attrs_t else df)["p_min_pu"],
                        p_max_pu=(dfs_t if "p_max_pu" in attrs_t else df)["p_max_pu"],
                        p_min_pu_annual=df["p_min_pu_annual"],
                        p_max_pu_annual=df["p_max_pu_annual"],
                        p_set_pu_annual=df["p_set_pu_annual"],
                        p_set=dfs_t["p_set"] if "p_set" in attrs_t else np.nan,
                        build_year=df["build_year"],
                        lifetime=df["lifetime"],
                        parent=df["parent"],
                        parent_ratio=df["parent_ratio"],
                    )
                    # In case efficiency is dynamic, still keep the static nominal efficiency
                    if "efficiency" in attrs_t:
                        network.links.loc[df.index, "efficiency"] = df["efficiency"]
                else:
                    network.add(
                        component,
                        df.index,
                        bus0=df["bus_output"],
                        bus1=df["bus_input"],
                        bus2=(
                            df["bus_output2"].fillna("") if "bus_output2" in df else ""
                        ),
                        area=df["area"],
                        area_from=df["area_from"],
                        carrier=df["carrier"],
                        technology=df["technology"],
                        qualifier=df["qualifier"],
                        aggregation=df["aggregation"],
                        p_nom=df["p_nom"],
                        p_nom_extendable=df["p_nom_extendable"],
                        p_nom_min=df["p_nom_min"],
                        p_nom_max=df["p_nom_max"],
                        efficiency=1
                        / (dfs_t if "efficiency" in attrs_t else df)["efficiency"],
                        efficiency2=(
                            -df["efficiency2"] / df["efficiency"]
                            if "efficiency2" in df
                            else np.nan
                        ),
                        length=df["length"],
                        variable_cost=-df["variable_cost"],
                        co2_cost=-df["co2_cost"],
                        marginal_cost=-df["marginal_cost"],
                        fixed_cost=df["fixed_cost"] * df["length"].fillna(1.0),
                        investment_cost=df["investment_cost"]
                        * df["length"].fillna(1.0),
                        annual_investment_cost=df["annual_investment_cost"]
                        * df["length"].fillna(1.0),
                        capital_cost=df["capital_cost"] * df["length"].fillna(1.0),
                        p_min_pu=-(dfs_t if "p_max_pu" in attrs_t else df)["p_max_pu"],
                        p_max_pu=-(dfs_t if "p_min_pu" in attrs_t else df)["p_min_pu"],
                        p_set=-dfs_t["p_set"] if "p_set" in attrs_t else np.nan,
                        p_min_pu_annual=-df["p_max_pu_annual"],
                        p_max_pu_annual=-df["p_min_pu_annual"],
                        p_set_pu_annual=-df["p_set_pu_annual"],
                        build_year=df["build_year"],
                        lifetime=df["lifetime"],
                        parent=df["parent"],
                        parent_ratio=df["parent_ratio"],
                    )
                    # In case efficiency is dynamic, still keep the static nominal efficiency
                    if "efficiency" in attrs_t:
                        network.links.loc[df.index, "efficiency"] = 1 / df["efficiency"]

                network.links = network.static(component).sort_index()

            elif component == "Line":

                # In case of lines, we should provide num_parallel instead of s_nom
                # s_nom (MVA) = sqrt(3) * i_nom (kA) * v_nom (kV) * num_parallel
                i_nom = df["line_type"].map(network.line_types["i_nom"])
                v_nom = df["bus_input"].map(network.buses["v_nom"])

                network.add(
                    component,
                    df.index,
                    bus0=df["bus_input"],
                    bus1=df["bus_output"],
                    area=df["area"],
                    area_from=df["area_from"],
                    carrier=df["carrier"],
                    technology=df["technology"],
                    qualifier=df["qualifier"],
                    aggregation=df["aggregation"],
                    type=df["line_type"],
                    num_parallel=df["s_nom"] / (np.sqrt(3) * i_nom * v_nom),
                    s_nom=df["s_nom"],
                    s_nom_extendable=df["s_nom_extendable"],
                    s_nom_min=df["s_nom_min"],
                    s_nom_max=df["s_nom_max"],
                    length=df["length"],
                    fixed_cost=df["fixed_cost"] * df["length"],
                    investment_cost=df["investment_cost"] * df["length"],
                    annual_investment_cost=df["annual_investment_cost"] * df["length"],
                    capital_cost=df["capital_cost"] * df["length"],
                    s_max_pu=df["s_max_pu"],
                    build_year=df["build_year"],
                    lifetime=df["lifetime"],
                    parent=df["parent"],
                    parent_ratio=df["parent_ratio"],
                )
                network.lines = network.static(component).sort_index()

            elif component == "Store":

                network.add(
                    component,
                    df.index,
                    bus=df["bus"],
                    area=df["area"],
                    carrier=df["carrier"],
                    technology=df["technology"],
                    qualifier=df["qualifier"],
                    aggregation=df["aggregation"],
                    e_nom=df["e_nom"],
                    e_nom_extendable=df["e_nom_extendable"],
                    e_nom_min=df["e_nom_min"],
                    e_nom_max=df["e_nom_max"],
                    e_min_pu=(dfs_t if "e_min_pu" in attrs_t else df)["e_min_pu"],
                    e_max_pu=(dfs_t if "e_max_pu" in attrs_t else df)["e_max_pu"],
                    standing_loss=df["standing_loss"],
                    variable_cost=df["variable_cost"],
                    co2_cost=df["co2_cost"],
                    marginal_cost=df["marginal_cost"],
                    fixed_cost=df["fixed_cost"],
                    investment_cost=df["investment_cost"],
                    annual_investment_cost=df["annual_investment_cost"],
                    capital_cost=df["capital_cost"],
                    build_year=df["build_year"],
                    lifetime=df["lifetime"],
                    parent=df["parent"],
                    parent_ratio=df["parent_ratio"],
                    e_cyclic=True,
                )
                network.stores = network.static(component).sort_index()

            # Exclude columns with all nans in p_set
            if "p_set" in dfs_t:
                network.dynamic(component)["p_set"] = network.dynamic(component)[
                    "p_set"
                ].dropna(axis=1, how="all")


def add_capacity_constraints(network, inputs, params):

    # This function assumes network.areas and network.carriers are already filled

    df_pot = inputs["capacity_addition_potentials"]
    df_pot = df_pot[df_pot["year"] == params["year"]].drop(columns="year")
    df_pot = df_pot[df_pot["carrier"].isin(network.carriers.index)]

    # Clear existing columns with capacity constraints
    nom_cols = network.areas.columns[network.areas.columns.str.startswith("nom_")]
    network.areas = network.areas.drop(columns=nom_cols)
    network.carriers["max_growth"] = np.inf

    for attr, df in df_pot.groupby("attribute"):
        if attr in ["nom_max", "nom_min"]:
            # Add columns with nominal capacity limits to network.areas
            df["column"] = df["attribute"] + "_" + df["carrier"]
            df = df.pivot(index="area", columns="column", values="value")
            network.areas = network.areas.join(df, how="left")
        elif attr == "max_growth":
            # Add max_growth column to network.carriers
            # Only constraint for the aggregation of domestic regions ("PL") is supported
            # During optimisation, carriers in foreign regions get area suffix
            assert (df["area"] == "PL").all()
            df = df[["carrier", "value"]].rename(columns={"value": "max_growth"})
            df = df.set_index("carrier").reindex(network.carriers.index).fillna(np.inf)
            network.carriers = network.carriers.drop(columns="max_growth").join(
                df, how="left"
            )


def add_energy_flow_constraints(network, inputs, params):

    df_flow = inputs["annual_energy_flows"]
    df_carr = inputs["technology_carrier_definitions"]

    if params["constrained_energy_flows"] == "none":
        return
    elif params["constrained_energy_flows"] == "all":
        df = df_flow
    else:
        df = df_flow[df_flow["carrier"].isin(params["constrained_energy_flows"])]

    df_carr_component = df_carr[["carrier", "component"]].drop_duplicates()
    # Important: we assume that all technologies within a carrier are represented by a single component type
    assert df_carr_component["carrier"].value_counts().max() == 1
    df = df.merge(df_carr_component, on="carrier", how="left")

    # If links are reverted, only outflow constraints are allowed
    if params["reverse_links"]:
        assert df.loc[df["component"] == "Link", "type"].eq("outflow").all()
    else:
        assert df.loc[df["component"] == "Link", "type"].eq("inflow").all()

    # Convert TWh to MWh (MtCO2 to tCO2)
    df["value"] = df["value"] * 1e6

    df["sense"] = df["parameter"].map(
        {
            "max_flow": "<=",
            "min_flow": ">=",
            "flow": "==",
        }
    )

    df = df[df["year"] == params["year"]].drop(columns="year")
    df["type"] = "operational_limit"
    # For subnational PL areas, use custom area-aware constraint
    domestic_prefix = "PL"
    df.loc[
        (df["area"] != domestic_prefix) & df["area"].str.startswith(domestic_prefix),
        "type",
    ] = "operational_limit_per_area"

    df = df.set_index("name")[["area", "carrier", "type", "sense", "value"]]

    co2_emissions = params.get("co2_emissions", None)
    if co2_emissions is not None:
        df.loc[f"PL CO2 emissions {params['year']}"] = [
            "PL",
            "co2_emissions",
            "custom_primary_energy_limit",
            "==",
            co2_emissions * 1e6,
        ]

    network.remove("GlobalConstraint", network.global_constraints.index)
    network.add(
        "GlobalConstraint",
        df.index,
        type=df["type"],
        investment_period=(
            np.nan if isinstance(params["year"], int) else params["year"][0]
        ),
        area=df["area"],
        carrier_attribute=df["carrier"],
        sense=df["sense"],
        constant=df["value"],
    )
