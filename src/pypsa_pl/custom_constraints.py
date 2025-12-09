import logging

import pandas as pd
from pyparsing import col
from xarray import DataArray
import numpy as np
from functools import reduce
from linopy.expressions import merge
from pypsa.descriptors import nominal_attrs
from pypsa.descriptors import get_switchable_as_dense as get_as_dense


def remove_annual_carrier_demand_constraints(n, carriers):
    for carrier in carriers:
        constraints = n.global_constraints[
            n.global_constraints["carrier_attribute"] == f"{carrier} final use"
        ].index
        constraints = ("GlobalConstraint-" + constraints).tolist()
        for constraint in constraints:
            if constraint in n.model.constraints:
                n.model.remove_constraints(constraint)


def define_operational_limit_per_area(n, sns):
    """
    Based on https://github.com/PyPSA/PyPSA/blob/v0.32.1/pypsa/optimization/global_constraints.py#L319-L384

    Defines operational limit constraints per area and carrier. It applies to generators and links.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    weightings = n.snapshot_weightings.loc[sns]
    glcs = n.global_constraints.query('type == "operational_limit_per_area"')

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    for name, glc in glcs.iterrows():
        snapshots = (
            sns
            if np.isnan(glc.investment_period)
            else sns[sns.get_loc(glc.investment_period)]
        )
        lhs = []
        rhs = glc.constant

        cond = "(carrier == @glc.carrier_attribute) & (area == @glc.area)"

        # generators
        gens = n.generators.query(cond)
        if not gens.empty:
            p = m["Generator-p"].loc[snapshots, gens.index]
            w = DataArray(weightings.generators[snapshots])
            if "dim_0" in w.dims:
                w = w.rename({"dim_0": "snapshot"})
            expr = (p * w).sum()
            lhs.append(expr)

        # links
        links = n.links.query(cond)
        if not links.empty:
            # Negative p0 is positve production at bus0
            p = -m["Link-p"].loc[snapshots, links.index]
            w = DataArray(weightings.generators[snapshots])
            if "dim_0" in w.dims:
                w = w.rename({"dim_0": "snapshot"})
            expr = (p * w).sum()
            lhs.append(expr)

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, rhs, f"GlobalConstraint-{name}")


def define_operational_limit_link(n, sns):
    """
    Based on https://github.com/PyPSA/PyPSA/blob/v0.32.1/pypsa/optimization/global_constraints.py#L319-L384

    Defines operational limit constraints. It limits the net production at bus0 of a link.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    weightings = n.snapshot_weightings.loc[sns]
    glcs = n.global_constraints.query('type == "operational_limit"')

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    for name, glc in glcs.iterrows():
        snapshots = (
            sns
            if np.isnan(glc.investment_period)
            else sns[sns.get_loc(glc.investment_period)]
        )
        lhs = []
        rhs = glc.constant

        # links
        links = n.links.query("carrier == @glc.carrier_attribute")
        if not links.empty:
            # Negative p0 is positve production at bus0
            p = -m["Link-p"].loc[snapshots, links.index]
            w = DataArray(weightings.generators[snapshots])
            if "dim_0" in w.dims:
                w = w.rename({"dim_0": "snapshot"})
            expr = (p * w).sum()
            lhs.append(expr)

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, rhs, f"GlobalConstraint-{name}")


def define_custom_primary_energy_limit(n, sns):
    """
    Based on https://github.com/PyPSA/PyPSA/blob/v0.32.1/pypsa/optimization/global_constraints.py#L241-L316
    The original version does not include links. This one includes generators and links.

    Defines primary energy constraints. It limits the byproducts of primary
    energy sources (defined by carriers) such as CO2.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    assert n.meta["reverse_links"]

    m = n.model
    weightings = n.snapshot_weightings.loc[sns]
    glcs = n.global_constraints.query('type == "custom_primary_energy_limit"')

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    for name, glc in glcs.iterrows():
        if np.isnan(glc.investment_period):
            sns_sel = slice(None)
        elif glc.investment_period in sns.unique("period"):
            sns_sel = sns.get_loc(glc.investment_period)
        else:
            continue

        lhs = []
        rhs = glc.constant
        emissions = n.carriers[glc.carrier_attribute][lambda ds: ds != 0]

        if emissions.empty:
            continue

        # generators
        gens = n.generators.query("carrier in @emissions.index")
        if not gens.empty:
            efficiency = get_as_dense(
                n, "Generator", "efficiency", snapshots=sns[sns_sel], inds=gens.index
            )
            em_pu = gens.carrier.map(emissions) / efficiency
            em_pu = em_pu.multiply(weightings.generators[sns_sel], axis=0)
            p = m["Generator-p"].loc[sns[sns_sel], gens.index]
            expr = (p * em_pu).sum()
            lhs.append(expr)

        # links
        links = n.links.query("carrier in @emissions.index")
        if not links.empty:
            efficiency = get_as_dense(
                n, "Link", "efficiency", snapshots=sns[sns_sel], inds=links.index
            )
            # Need to invert the efficiency timeseries, to get consumption at bus1
            efficiency = 1 / efficiency
            em_pu = links.carrier.map(emissions) / efficiency
            em_pu = em_pu.multiply(weightings.generators[sns_sel], axis=0)
            # Negative p0 is positive production at bus0
            p = -m["Link-p"].loc[sns[sns_sel], links.index]
            expr = (p * em_pu).sum()
            lhs.append(expr)

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, rhs, f"GlobalConstraint-{name}")


def define_nominal_constraints_per_area_carrier(n, sns):
    """
    Based on https://github.com/PyPSA/PyPSA/blob/v0.32.1/pypsa/optimization/global_constraints.py#L94-L175
    Compared to PyPSA's default capacity constraint per bus and carrier it
    (1) works for p_nom at bus0 of links
    (2) includes fixed capacities

    Set a capacity limit for assets of the same carrier at the same area.
    The function searches for columns in the `area` dataframe matching
    the pattern "nom_{min/max}_{carrier}". In case the constraint should only
    be defined for one investment period, the column name can be constructed
    according to "nom_{min/max}_{carrier}_{period}" where period must be in
    `n.investment_periods`.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    cols = n.areas.columns[n.areas.columns.str.startswith("nom_")]
    areas = n.areas.index[n.areas[cols].notnull().any(axis=1)].rename(
        "Area-nom_min_max"
    )

    for col in cols:
        msg = (
            f"Area column '{col}' has invalid specification and cannot be "
            "interpreted as constraint, must match the pattern "
            "`nom_{min/max}_{carrier}` or `nom_{min/max}_{carrier}_{period}`"
        )
        if col.startswith("nom_min_"):
            sign = ">="
        elif col.startswith("nom_max_"):
            sign = "<="
        else:
            logging.warning(msg)
            continue
        remainder = col[len("nom_max_") :]
        if remainder in n.carriers.index:
            carrier = remainder
            period = None
        elif isinstance(n.snapshots, pd.MultiIndex):
            carrier, period = remainder.rsplit("_", 1)
            period = int(period)
            if carrier not in n.carriers.index or period not in sns.unique("period"):
                logging.warning(msg)
                continue
        else:
            logging.warning(msg)
            continue

        lhs = []
        rhs = n.areas.loc[areas, col]

        for c, attr in [
            ("Generator", "p_nom"),
            ("Link", "p_nom"),
            ("Store", "e_nom"),
        ]:
            var = f"{c}-{attr}"
            dim = f"{c}-ext"
            df = n.static(c)

            # *** Special case of "PL" area constraints ***
            # Apply the "PL" area constraints to the summed capacities from all areas with "PL" prefix
            # This requires that for components in areas with "PL" prefix we have to cerate a duplicate row with area replaced by "PL"
            df_pl = df[df["area"].str.startswith("PL") & (df["area"] != "PL")].copy()
            if not df_pl.empty:
                df_pl["area"] = "PL"
                df = pd.concat([df, df_pl], axis=0)
            # ***

            non_ext_i = n.get_non_extendable_i(c).intersection(
                df.index[df.carrier == carrier]
            )
            if period is not None:
                non_ext_i = non_ext_i[n.get_active_assets(c, period)[non_ext_i]]
            # Group p_nom of components by area and sum
            nom_fixed = df.loc[non_ext_i].groupby("area")[attr].sum()
            rhs -= nom_fixed.reindex(rhs.index, fill_value=0)

            # Extendable capacities

            # *** Since mapping is non-unique we need to group by area in two steps
            for sel in [df["area"] == "PL", df["area"] != "PL"]:
                df_sel = df[sel]
                ext_i = (
                    n.get_extendable_i(c)
                    .intersection(df_sel.index[df_sel.carrier == carrier])
                    .rename(dim)
                )
                if period is not None:
                    ext_i = ext_i[n.get_active_assets(c, period)[ext_i]]

                if ext_i.empty:
                    continue

                # Create a mapping from index to area
                areamap = df_sel.loc[ext_i, "area"].rename(areas.name).to_xarray()

                # Group by area and sum the p_nom
                expr = (
                    m[var]
                    .loc[ext_i]
                    .groupby(areamap)
                    .sum()
                    # .reindex({areas.name: areas})
                )
                lhs.append(expr)
            # ***

        if not lhs:
            continue

        if col.startswith("nom_max") and (rhs < 0).any():
            logging.warning(f"Infeasible {col} constraint")
            logging.warning(rhs[rhs < 0])
            logging.warning("Setting RHS to 0")
            rhs.loc[rhs < 0] = 0

        lhs = merge(lhs)

        # *** Apply the constraint only if there are relevant extendable capacities
        index_lhs = lhs.coords[areas.name].values
        index_rhs = rhs[rhs.notnull()].index
        index = index_rhs.intersection(index_lhs)
        lhs = lhs.reindex({areas.name: index})
        rhs = rhs.reindex(index)
        # ***

        n.model.add_constraints(lhs, sign, rhs, name=f"Area-{col}")


def define_annual_capacity_utilisation_constraints(n, sns):

    # Those constraints at annual level are only effective if no constraint at snapshot level is defined
    # E.g. if time-dependent p_min_pu == p_max_pu are defined, the p_set_pu_annual is not effective

    m = n.model

    for suffix, sign in [("set", "=="), ("max", "<="), ("min", ">=")]:
        attr = f"p_{suffix}_pu"
        attr_annual = f"{attr}_annual"
        for c in ["Generator", "Link"]:
            components = n.static(c)
            components_t = n.dynamic(c)

            if attr_annual not in components.columns:
                continue

            if suffix == "set":
                components = components[components[attr_annual].notna()]
                # Exclude components with time-dependent p_min_pu == p_max_pu
                if "p_min_pu" in components_t and "p_max_pu" in components_t:
                    intersect = components_t["p_min_pu"].columns.intersection(
                        components_t["p_max_pu"].columns
                    )
                    equal = (
                        components_t["p_min_pu"][intersect]
                        == components_t["p_max_pu"][intersect]
                    ).all(axis=0)
                    exclude = equal[equal].index
                    components = components[~components.index.isin(exclude)]
            else:
                components = components.query(
                    f"{attr_annual} {sign.replace('=', '')} {attr}"
                )
                # Exclude components with time-dependent p_(min/max)_pu
                if attr in components_t:
                    exclude = components_t[attr].columns
                    components = components[~components.index.isin(exclude)]
            # In any case exclude components with time-dependent p_set
            if "p_set" in components_t:
                exclude = components_t["p_set"].columns
                components = components[~components.index.isin(exclude)]
            if components.empty:
                continue

            for extendable, df in components.groupby("p_nom_extendable"):
                if df.empty:
                    continue
                dim_name = f"{c}-{'ext' if extendable else 'fix'}-{attr_annual}"
                p_annual = (
                    m.variables[f"{c}-p"]
                    .loc[:, df.index]
                    .sum("snapshot")
                    .rename({c: dim_name})
                )
                utilisation = df[attr_annual].rename_axis(dim_name).to_xarray()
                N = len(n.snapshots)
                lhs = p_annual
                rhs = 0
                if extendable:
                    p_nom = (
                        m.variables[f"{c}-p_nom"]
                        .loc[df.index]
                        .rename({f"{c}-ext": dim_name})
                    )
                    lhs -= p_nom * utilisation * N
                else:
                    p_nom = df["p_nom"].rename_axis(dim_name).to_xarray()
                    rhs += p_nom * utilisation * N
                m.add_constraints(
                    lhs,
                    sign,
                    rhs,
                    f"{dim_name}-capacity_utilisation_constraint",
                )


def define_parent_children_capacity_constraints(n, sns):

    reverse_links = n.meta["reverse_links"]

    # Fix ratio of parent and children capacities
    # This constraint is effective only if both parent and child capacities are extendable

    m = n.model

    nom_attrs = {
        "Generator": "p_nom",
        "Link": "p_nom",
        "Store": "e_nom",
    }
    dim_name = "Component-ext-child"

    # Gather all nominal capacities of potential parent and children components
    nom = []
    for c, nom_attr in nom_attrs.items():
        if f"{c}-{nom_attr}" not in m.variables:
            continue
        expr = m.variables[f"{c}-{nom_attr}"].rename({f"{c}-ext": dim_name})
        if (c == "Link") and not reverse_links:
            eff = (
                n.links.loc[expr.coords[dim_name].values, "efficiency"]
                .rename_axis(dim_name)
                .to_xarray()
            )
            expr *= eff
        expr *= 1.0
        nom.append(expr)
    if len(nom) == 0:
        return
    nom = merge(nom, dim=dim_name)

    # Identify children and formulate constraints
    lhs = []
    for c, nom_attr in nom_attrs.items():
        components = n.static(c)
        children = components[
            components[f"{nom_attr}_extendable"]
            & (components[f"{nom_attr}_max"] > components[f"{nom_attr}_min"])
            & components["parent"].isin(nom.coords[dim_name].values)
        ].index

        if children.empty:
            continue

        nom_child = nom.loc[children]
        nom_parent = nom.loc[components.loc[children, "parent"].values].assign_coords(
            {dim_name: children.rename(dim_name)}
        )
        parent_ratio = (
            components.loc[children, "parent_ratio"].rename_axis(dim_name).to_xarray()
        )
        expr = nom_child - parent_ratio * nom_parent
        lhs.append(expr)

    if len(lhs) == 0:
        return

    lhs = merge(lhs, dim=dim_name)

    m.add_constraints(
        lhs, "==", 0, name=f"{dim_name}-parent_children_capacity_constraint"
    )


def define_fixed_ratios_to_space_heating(n, sns):

    assert n.meta["reverse_links"]

    # DO NOT CONSTRAIN OTHER HEATING LINKS
    # The model needs the flexibility to allocate the constant production of industrial CHPs
    # TODO: think of long-term solution
    carriers = [
        "water heating",
        # "other heating",
    ]

    # (1) Calculate ratios of water/other heating to space heating annual flows

    ratios = {}

    annual_flows = {
        carrier: n.global_constraints[
            n.global_constraints["carrier_attribute"] == f"{carrier} final use"
        ]
        .groupby("area")["constant"]
        .sum()
        for carrier in ["space heating", "water heating", "other heating"]
    }
    annual_flows["total heating"] = reduce(
        lambda a, b: a.add(b, fill_value=0), annual_flows.values()
    )
    if (annual_flows["total heating"] == 0).all():
        return

    # space heating demand reduction due to building retrofitting
    # building retrofits need to be exogenously specified
    retrofits = n.generators[n.generators["carrier"] == "building retrofits"]
    assert (retrofits["p_nom_extendable"] == False).all()
    avoided_space_heating_demand = (
        n.generators[n.generators["carrier"] == "building retrofits"]
        .groupby("area")["p_nom"]
        .sum()
        * n.meta["heat_capacity_utilisation"]
        * 8760
    )

    # (1a) Water heating - applies to both centralised and decentralised heating

    ratios["water heating"] = annual_flows["water heating"] / (
        annual_flows["space heating"].add(-avoided_space_heating_demand, fill_value=0)
    )

    assert ratios["water heating"].notna().all()

    # TODO: the code below is not area-aware; however it is not needed if other heating ratio is not fixed
    # (1b) Other heating - applies to centralised heating only

    # If centralised heating share is exogenously specified, use it
    # Otherwise, calculate it from district heating capacity and utilisation
    # centralised_heating_share = n.meta.get("centralised_heating_share", None)
    # if centralised_heating_share is None:
    #     centralised_heating = (
    #         n.links.loc[n.links["carrier"] == "district heating", "p_nom"].sum()
    #         * n.meta["heat_capacity_utilisation"]
    #         * 8760
    #     )
    #     centralised_heating_share = centralised_heating / annual_flows["total heating"]

    # centralised_space_and_water_heating = (
    #     centralised_heating_share * annual_flows["total heating"]
    #     - annual_flows["other heating"]
    # )
    # centralised_space_heating = centralised_space_and_water_heating / (
    #     1 + ratios["water heating"]
    # )
    # ratios["other heating"] = annual_flows["other heating"] / centralised_space_heating

    # (2) Apply fixed ratios to heating links

    m = n.model

    if "Link-p_nom" not in m.variables:
        return

    p_nom = m.variables["Link-p_nom"]

    parent_techs = {
        "water heating": [
            "centralised space heating",
            "decentralised space heating",
        ],
        # "other heating": [
        #     "centralised space heating",
        # ],
    }

    for carrier in carriers:

        p_set_pu_annual_parents = n.meta["space_heating_utilisation"]
        p_set_pu_annual_children = n.meta[f"{carrier.replace(' ', '_')}_utilisation"]

        for area, ratio in ratios[carrier].items():

            parents = n.links[
                n.links["technology"].isin(parent_techs[carrier])
                & n.links["p_nom_extendable"]
                & (n.links["area"] == area)
            ].index

            if parents.empty:
                continue

            def rename_parents_to_children(parents):
                return parents.str.replace("space heating", carrier)

            children = rename_parents_to_children(parents)
            assert children.isin(n.links.index).all()

            p_nom_children = p_nom.loc[children]
            p_nom_parents = p_nom.loc[parents]

            p_nom_parents = p_nom_parents.assign_coords(
                {
                    "Link-ext": rename_parents_to_children(
                        p_nom_parents.coords["Link-ext"]
                    )
                }
            )

            lhs = (
                p_set_pu_annual_children * p_nom_children
                - ratio * p_set_pu_annual_parents * p_nom_parents
            )

            coord_name = (
                f"Link-ext-{area.replace(' ', '_')}-{carrier.replace(' ', '_')}"
            )
            lhs = lhs.rename({"Link-ext": coord_name})

            m.add_constraints(
                lhs, "==", 0, name=f"{coord_name}-fixed_ratio_to_space_heating"
            )

            logging.info(
                f"Fixed ratio of {carrier} to space heating in {area}: {ratio:.2f}"
            )

        logging.info(f"Removing annual flow constraint for {carrier}")
        remove_annual_carrier_demand_constraints(n, carriers=[carrier])


def define_heating_capacity_utilisation_constraints(n, sns):

    assert n.meta["reverse_links"]

    buses = n.buses[
        n.buses["carrier"].isin(
            [
                "heat decentralised",
                "heat centralised out",
            ]
        )
    ].index

    if buses.empty:
        return

    n_all = len(buses)

    # (1) Find heating demand links connected to heat buses
    demand = (
        n.links[
            n.links["bus1"].isin(buses)
            & n.links["carrier"].str.endswith(
                ("space heating", "water heating", "other heating")
            )
        ].rename(columns={"bus1": "bus"})
    )[["bus", "carrier", "p_nom_extendable"]]

    # If heating demand links are non-extendable, this constraint cannot be applied
    if not demand["p_nom_extendable"].all():
        return

    demand = demand.drop(columns="p_nom_extendable").reset_index(names="name")
    demand["carrier"] = (
        demand["carrier"]
        .str.replace("decentralised ", "")
        .str.replace("centralised ", "")
    )
    demand = demand.pivot(index="bus", columns="carrier", values="name")

    # (2) Find supplying generators and links connected to heat buses
    generators = n.generators[n.generators["bus"].isin(buses)].assign(
        component="Generator"
    )
    links = (
        n.links[n.links["bus0"].isin(buses)]
        .rename(columns={"bus0": "bus"})
        .assign(component="Link")
    )
    supply = pd.concat([generators, links])[
        [
            "bus",
            "p_nom",
            "technology",
            "carrier",
            "qualifier",
            "component",
            "p_nom_extendable",
        ]
    ].reset_index(names="name")

    # Exception for heat pump bus - drop resistive heater and heat storage small acting as secondary sources from supply technologies
    # TODO: find a non-hard coded solution
    supply = supply[
        ~(
            supply["bus"].str.contains("heat pump")
            & (
                supply["technology"].isin(
                    ["resistive heater small", "heat storage small discharge"]
                )
            )
        )
    ]

    n_extendable = (
        supply[["bus", "p_nom_extendable"]].drop_duplicates()["p_nom_extendable"].sum()
    )

    # (3) Enforce utilisation of heat supplying component only if it has a single technology
    # This excludes e.g. heat pump combined with heat storage
    # In case of centralised heating, supply is provided by a single district heating link

    n_supply_techs = supply.groupby("bus")["technology"].nunique()
    buses = n_supply_techs[n_supply_techs == 1].index

    supply = supply.set_index("bus").loc[buses]
    demand = demand.loc[buses]

    # (4) Calculate LHS and RHS for each bus

    m = n.model

    lhs = []
    rhs = []

    # (4a) LHS: sum of demand link capacities multiplied by relevant utilisation factors
    p_nom = m.variables["Link-p_nom"]
    for carrier in ["space heating", "water heating", "other heating"]:
        if carrier not in demand.columns:
            continue
        is_not_na = demand[carrier].notna()
        expr = (
            p_nom.loc[demand.loc[is_not_na, carrier].values]
            * n.meta[f"{carrier.replace(' ', '_')}_utilisation"]
        )
        expr = expr.rename({"Link-ext": "Bus-heat"}).assign_coords(
            {"Bus-heat": demand[is_not_na].index.rename("Bus-heat")}
        )
        lhs.append(expr)

    # (4b) LHS or RHS: sum of heat supply capacities multiplied by heating capacity utilisation factor
    for c, components in supply.groupby("component"):
        for is_extendable, supply in components.groupby("p_nom_extendable"):
            if is_extendable:
                p_nom = m.variables[f"{c}-p_nom"]
                expr = (
                    -p_nom.loc[supply["name"].values]
                    * n.meta["heat_capacity_utilisation"]
                )
                expr = expr.rename({f"{c}-ext": "Bus-heat"}).assign_coords(
                    {"Bus-heat": supply.index.rename("Bus-heat")}
                )
                if len(supply.index) != supply.index.nunique():
                    expr = expr.groupby("Bus-heat").sum()
                lhs.append(expr)
            else:
                value = supply["p_nom"] * n.meta["heat_capacity_utilisation"]
                value = value.rename_axis("Bus-heat")
                value = value.groupby(value.index).sum()
                rhs.append(value)

    lhs = merge(lhs)
    if len(rhs) == 0:
        rhs = 0
    else:
        rhs = pd.concat(rhs, axis=1).sum(axis=1)
        # Make sure that RHS has the same index as LHS
        # If necessary, fill missing values with 0
        rhs = rhs.reindex(lhs.coords["Bus-heat"].values).fillna(0).to_xarray()

    m.add_constraints(lhs, "==", rhs, name="Bus-heat-heating_capacity_utilisation")

    logging.info(f"Fixed heat flows at {len(buses)} out of {n_all} heat buses")
    if (len(buses) == n_all) and (n_extendable == 0):
        logging.info(
            "Removing annual flow constraint for space heating because no heating capacity is extendable"
        )
        remove_annual_carrier_demand_constraints(n, carriers=["space heating"])


def define_centralised_heating_share_constraint(n, sns):

    year = n.meta["year"]

    assert n.meta["reverse_links"]

    centralised_heating_shares = n.meta.get("centralised_heating_shares", None)
    if centralised_heating_shares is None:
        return

    centralised_heating_shares = pd.Series(
        centralised_heating_shares, name="centralised heating share"
    )

    buses = n.buses[n.buses["carrier"] == "heat centralised out"].index
    if buses.empty:
        return

    # Find heating demand links connected to centralised heat buses
    demand = (n.links[n.links["bus1"].isin(buses)].rename(columns={"bus1": "bus"}))[
        ["bus", "area", "carrier", "p_nom_extendable"]
    ]

    # If heating demand links are non-extendable, this constraint cannot be applied
    if not demand["p_nom_extendable"].all():
        return

    areamap = (
        demand["area"]
        .rename("Area-centralised_heating_share")
        .rename_axis("Link-ext")
        .to_xarray()
    )

    # (1) Calculate LHS as the total centralised heating output
    lhs = 0

    demand = demand.drop(columns="p_nom_extendable").reset_index(names="name")
    demand["carrier"] = demand["carrier"].str.replace("centralised ", "")
    demand = demand.pivot(index="bus", columns="carrier", values="name")

    m = n.model
    p_nom = m.variables["Link-p_nom"]
    for carrier in [
        "space heating",
        "water heating",
        "other heating",
    ]:
        index = demand[carrier].values
        expr = (
            p_nom.loc[index].groupby(areamap.loc[index]).sum()
            * n.meta[f"{carrier.replace(' ', '_')}_utilisation"]
        )
        lhs += expr

    # (1a) Subtract avoided space heating demand due to building retrofitting from RHS
    rhs = 0
    retrofits = n.generators[
        (n.generators["carrier"] == "building retrofits")
        & (n.generators["qualifier"] == "heat centralised")
        & (n.generators["build_year"] <= year)
        & (n.generators["build_year"] + n.generators["lifetime"] > year)
    ]
    # building retrofits need to be exogenously specified
    assert (retrofits["p_nom_extendable"] == False).all()
    if not retrofits.empty:
        rhs -= (
            retrofits.groupby("area")["p_nom"].sum()
            * n.meta["heat_capacity_utilisation"]
        )

    # (2) Calculate RHS as the total heating demand times the centralised heating share

    annual_flows = {
        carrier: n.global_constraints[
            n.global_constraints["carrier_attribute"] == f"{carrier} final use"
        ]
        .groupby("area")["constant"]
        .sum()
        for carrier in ["space heating", "water heating", "other heating"]
    }

    # *** If we go to voivodeship level, we cannot just sum all the flows, as we leave the model freedom to assign other heating flows to voivodeships
    # annual_flows["total heating"] = sum(annual_flows.values())

    # Centralised heating needs to cover at least the other heating demand
    # min_global_centralised_heating_flow = (
    #     annual_flows["other heating"].sum()
    # )

    # assert (centralised_heating_shares * annual_flows["total"]).sum() >= min_global_centralised_heating_flow
    # ***

    # Divide by 8760 to have the same units as the LHS

    rhs += (
        centralised_heating_shares
        * (annual_flows["space heating"] + annual_flows["water heating"])
        / 8760
    )

    # (2a) Subtract endogenously determined other heating flows from the LHS
    index = demand["other heating"].values
    other_heating_annual_flow = (
        p_nom.loc[index].groupby(areamap.loc[index]).sum()
        * n.meta["other_heating_utilisation"]
        * 8760
    )
    lhs -= centralised_heating_shares * other_heating_annual_flow / 8760

    # (3) Add constraint
    rhs.index.name = "Area-centralised_heating_share"

    m.add_constraints(lhs, "==", rhs, name="Area-centralised_heating_share")


def define_non_heating_capacity_utilisation_constraints(n, sns):

    assert n.meta["reverse_links"]

    for carrier in ["light vehicle mobility"]:

        dim_name = "Bus-" + carrier.replace(" ", "_")

        # Identify intermediate (i.e. having a qualifier) buses of a given carrier
        buses = n.buses[
            (n.buses["carrier"] == carrier) & (n.buses["qualifier"].fillna("") != "")
        ].index

        if buses.empty:
            return

        n_all = len(buses)

        # (1) Find demand links connected to intermediate buses
        demand = (
            n.links[
                n.links["bus1"].isin(buses) & (n.links["carrier"] == carrier)
            ].rename(columns={"bus1": "bus"})
        )[["bus", "carrier", "p_nom_extendable"]]

        # If demand links are non-extendable, this constraint cannot be applied
        if not demand["p_nom_extendable"].all():
            return

        demand = demand.drop(columns="p_nom_extendable").reset_index(names="name")

        # (2) Find supplying generators and links connected to intermediate buses
        generators = n.generators[n.generators["bus"].isin(buses)].assign(
            component="Generator"
        )
        links = (
            n.links[n.links["bus0"].isin(buses)]
            .rename(columns={"bus0": "bus"})
            .assign(component="Link")
        )
        supply = pd.concat([generators, links])[
            ["bus", "p_nom", "technology", "carrier", "component", "p_nom_extendable"]
        ].reset_index(names="name")

        n_extendable = (
            supply[["bus", "p_nom_extendable"]]
            .drop_duplicates()["p_nom_extendable"]
            .sum()
        )

        # (3) Enforce utilisation of carrier supplying component only if it contains a single technology
        n_supply_techs = supply.groupby("bus")["technology"].nunique()
        buses = n_supply_techs[n_supply_techs == 1].index

        supply = supply.set_index("bus").loc[buses]
        demand = demand.set_index("bus").loc[buses]

        # (4) Calculate LHS and RHS for each bus

        m = n.model

        lhs = []
        rhs = []

        # (4a) LHS: sum of demand link capacities multiplied by relevant utilisation factors
        p_nom = m.variables["Link-p_nom"]
        expr = (
            p_nom.loc[demand["name"].values]
            * n.meta[f"{carrier.replace(' ', '_')}_utilisation"]
        )
        expr = expr.rename({"Link-ext": dim_name}).assign_coords(
            {dim_name: demand.index.rename(dim_name)}
        )
        lhs.append(expr)

        # (4b) LHS or RHS: sum of supply capacities multiplied by relevant capacity utilisation factor
        for c, components in supply.groupby("component"):
            for is_extendable, supply in components.groupby("p_nom_extendable"):
                if is_extendable:
                    p_nom = m.variables[f"{c}-p_nom"]
                    expr = (
                        -p_nom.loc[supply["name"].values]
                        * n.meta[f"{carrier.replace(' ', '_')}_utilisation"]
                    )
                    expr = expr.rename({f"{c}-ext": dim_name}).assign_coords(
                        {dim_name: supply.index.rename(dim_name)}
                    )
                    if len(supply.index) != supply.index.nunique():
                        expr = expr.groupby(dim_name).sum()
                    lhs.append(expr)
                else:
                    value = (
                        supply["p_nom"]
                        * n.meta[f"{carrier.replace(' ', '_')}_utilisation"]
                    )
                    value = value.rename_axis(dim_name)
                    value = value.groupby(value.index).sum()
                    rhs.append(value)

        lhs = merge(lhs)
        if len(rhs) == 0:
            rhs = 0
        else:
            rhs = pd.concat(rhs, axis=1).sum(axis=1)
            # Make sure that RHS has the same index as LHS
            # If necessary, fill missing values with 0
            rhs = rhs.reindex(lhs.coords[dim_name].values).fillna(0).to_xarray()

        m.add_constraints(lhs, "==", rhs, name=f"{dim_name}-capacity_utilisation")

        logging.info(
            f"Fixed {carrier} flows at {len(buses)} out of {n_all} {carrier} buses"
        )
        if (len(buses) == n_all) and (n_extendable == 0):
            logging.info(
                f"Removing annual flow constraint for {carrier} because no {carrier} capacity is extendable"
            )
            remove_annual_carrier_demand_constraints(n, carriers=[carrier])


def define_minimum_bev_charge_level_constraint(n, sns):
    # not used at the moment
    bev_batteries = n.stores[n.stores["technology"] == "BEV battery"]
    bev_batteries_i = bev_batteries.index

    if bev_batteries_i.empty:
        return

    hour = n.meta.get("minimum_bev_charge_hour", 6)
    e_min_pu = n.meta.get("minimum_bev_charge_level", 0.75)

    n_levels = n.snapshots.nlevels  # 1 for index, 2 for multiindex
    constraint_snapshots_i = n.snapshots[
        pd.to_datetime(n.snapshots.get_level_values(n_levels - 1)).hour == hour
    ]

    m = n.model

    lhs = 0
    rhs = []

    # (1) LHS: charge level of BEV batteries at the specified hour
    e_t = (
        m["Store-e"]
        .loc[constraint_snapshots_i, bev_batteries_i]
        .rename({"Store": "Store-BEV_battery"})
    )
    lhs += e_t

    # (2) LHS or RHS: minimum charge level of BEV batteries
    for e_nom_extendable, df in bev_batteries.groupby("e_nom_extendable"):
        if e_nom_extendable:
            e_nom = (
                m.variables["Store-e_nom"]
                .loc[df.index]
                .rename({"Store-ext": "Store-BEV_battery"})
            )
            lhs -= e_min_pu * e_nom
        else:
            e_nom = df["e_nom"].rename_axis("Store-BEV_battery")
            rhs.append(e_min_pu * e_nom)

    if len(rhs) == 0:
        rhs = 0
    else:
        rhs = pd.concat(rhs, axis=1).sum(axis=1)
        # Make sure that RHS has the same index as LHS
        # If necessary, fill missing values with 0
        rhs = rhs.reindex(lhs.coords["Store-BEV_battery"].values).fillna(0).to_xarray()

    m.add_constraints(
        lhs, ">=", rhs, name=f"Store-BEV_battery-min_bev_charge_level_at_{hour}h"
    )


def define_minimum_synchronous_generation(n, sns):

    p_min_synchronous = n.meta.get("p_min_synchronous", 0)
    synchronous_carriers = n.meta.get("synchronous_carriers", [])
    if (p_min_synchronous == 0) or (len(synchronous_carriers) == 0):
        return

    assert n.meta["reverse_links"]

    buses = n.buses[n.buses["carrier"] == "electricity in"].index
    if buses.empty:
        return

    lhs = 0
    m = n.model
    # Find components supplying to electricity buses (Stores not included)
    for component, bus in [("Generator", "bus"), ("Link", "bus0")]:
        df = n.static(component)
        df = df[df[bus].isin(buses) & df["carrier"].isin(synchronous_carriers)]
        if component == "Generator":
            sign = 1
            df = df[df["sign"] > 0]
        elif component == "Link":
            # because reverse_links is True
            sign = -1
        p = sign * m[f"{component}-p"].loc[sns, df.index].sum(component)
        lhs += p

    m.add_constraints(
        lhs, ">=", p_min_synchronous, name="synchronous_generation_constraint"
    )


def define_proportional_expansion_constraint(n, sns, allowed_ratio_deviation=0.01):

    # Expansion of capacities in areas that are not macroareas are proprotional to unused potential

    proportional_expansion = n.meta.get("proportional_expansion", None)
    if proportional_expansion is None:
        return

    m = n.model
    # Constraint works for areas of "PL *" format
    df_nom_max = n.areas.loc[
        n.areas.index.str.startswith("PL "),
        n.areas.columns.str.startswith("nom_max_"),
    ]
    name = "Area-proportional_expansion"

    for carrier in proportional_expansion:

        if f"nom_max_{carrier}" not in df_nom_max.columns:
            continue

        nom_max = df_nom_max[f"nom_max_{carrier}"]
        # We need potentials per each area
        if nom_max.isna().any():
            continue

        nom_existing = 0
        nom_ext = 0

        for c, attr in [
            ("Generator", "p_nom"),
            ("Link", "p_nom"),
            ("Store", "e_nom"),
        ]:
            var = f"{c}-{attr}"
            dim = f"{c}-ext"
            df = n.static(c)

            sel = df.carrier == carrier

            non_ext_i = n.get_non_extendable_i(c).intersection(df.index[sel])
            nom_existing += (
                df.loc[non_ext_i]
                .groupby("area")[attr]
                .sum()
                .reindex(nom_max.index, fill_value=0)
            )

            ext_i = n.get_extendable_i(c).intersection(df.index[sel]).rename(dim)
            if ext_i.empty:
                continue

            areamap = df.loc[ext_i, "area"].rename(name).to_xarray()
            nom_ext += m[var].loc[ext_i].groupby(areamap).sum()

        nom_available = (nom_max - nom_existing).rename(name).rename_axis(name)

        if (nom_available < 0).any():
            logging.warning(
                f"Potential of {carrier} in some areas is exceeded, setting potential to 0."
            )
            nom_available.loc[nom_available < 0] = 0

        ratios = nom_available / nom_available.sum()
        ratios_min = np.maximum(ratios - allowed_ratio_deviation, 0)
        ratios_max = np.minimum(ratios + allowed_ratio_deviation, 1)

        # # Define reference area for normalisation as the one with the largest potential
        # area_ref = nom_available.idxmax()
        # # Set the ratios min/max of capacities in other areas to the reference area
        # ratios_to_ref_min = ratios_min / ratios_max[area_ref]
        # ratios_to_ref_max = ratios_max / ratios_min[area_ref]

        nom_ext_sum = nom_ext.sum()

        for sign, ratios in [
            (">=", ratios_min),
            ("<=", ratios_max),
        ]:
            lhs = nom_ext - ratios.to_xarray() * nom_ext_sum
            # Drop constraints for reference area and areas with ratio zero or one
            index = ratios[
                (ratios != 0) & (ratios != 1)
                # & (ratios.index != area_ref)
            ].index
            lhs = lhs.reindex({name: index})

            m.add_constraints(
                lhs,
                sign,
                0,
                name=f"Area-proportional_expansion-{'min' if sign == '>=' else 'max'}-{carrier}",
            )
