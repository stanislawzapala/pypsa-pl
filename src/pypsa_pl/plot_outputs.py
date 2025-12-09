import pandas as pd
import numpy as np
import logging

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from math import ceil
import pypsa

from pypsa_pl.config import data_dir
from pypsa_pl.plots import plot_bar, plot_area, plot_line
from pypsa_pl.plots import mpl_style, get_order_and_colors
from pypsa_pl.process_output_network import (
    calculate_statistics,
    calculate_fuel_consumption,
    calculate_co2_emissions,
    calculate_capex,
    calculate_opex,
    calculate_output_capacities,
    calculate_input_capacities,
    calculate_storage_capacities,
    calculate_output_capacity_additions,
    calculate_grid_capacities,
    calculate_grid_capacity_additions,
    calculate_input_capacity_additions,
    calculate_storage_capacity_additions,
    calculate_flows,
    calculate_energy_balance_at_peak_load,
    calculate_curtailed_vres_energy,
    calculate_marginal_prices,
)


def get_label_threshold(ylim, figsize, default=0):
    if ylim is not None:
        return (ylim[1] - ylim[0]) / figsize[1] * 0.085
    else:
        return default


# Annual quantities


def plot_installed_capacities(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=["electricity in", "electricity out"],
    carrier_name="electricity",
    bus_qualifiers=None,
    capacity_type="generation",
    x_var="year",
    cat_var="carrier",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
):

    if carrier_name is None:
        carrier_name = (
            bus_carriers if isinstance(bus_carriers, str) else bus_carriers[0]
        )

    if network is not None:
        if capacity_type == "generation":
            # Output capacity, e.g. electrical capacity of a power plant
            df = calculate_output_capacities(
                network, bus_carriers, bus_qualifiers=bus_qualifiers
            )
        elif capacity_type == "consumption":
            # Input capacity, e.g. electrical capacity of an electrolyser
            df = calculate_input_capacities(
                network, bus_carriers, bus_qualifiers=bus_qualifiers
            )
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = (df["value"] / 1e3).round(2)
        # Exclude virtual components and grids
        df = df[
            ~df[cat_var].isin(
                ["hydrogen", "light vehicle mobility", "electricity grid"]
            )
            & ~df[cat_var].str.startswith(
                ("transformation", "connection", "distribution")
            )
        ]

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title=f"Installed {carrier_name} {capacity_type} capacity [GW]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 3.5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, df


def plot_capacity_additions(
    network=None,
    df=None,
    run_name=None,
    bus_carriers="electricity",
    carrier_name="electricity",
    bus_qualifiers=None,
    capacity_type="generation",
    x_var="year",
    cat_var="carrier",
    ylim=None,
    figsize=(4, 4),
    make_fig=True,
):

    if carrier_name is None:
        carrier_name = (
            bus_carriers if isinstance(bus_carriers, str) else bus_carriers[0]
        )

    if network is not None:

        if capacity_type == "generation":
            # Output capacity, e.g. electrical capacity of a power plant
            df = calculate_output_capacity_additions(
                network, bus_carriers, bus_qualifiers=bus_qualifiers
            )
        elif capacity_type == "consumption":
            # Input capacity, e.g. electrical capacity of an electrolyser
            df = calculate_input_capacity_additions(
                network, bus_carriers, bus_qualifiers=bus_qualifiers
            )
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = (df["value"] / 1e3).round(2)
        # Exclude virtual components
        df = df[~df[cat_var].isin(["hydrogen", "light vehicle mobility"])]

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )

    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title=f"{carrier_name.capitalize()} {capacity_type} capacity additions [GW]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_storage_capacities(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=["battery large electricity", "hydro PSH electricity"],
    carrier_name="electricity",
    bus_qualifiers=None,
    x_var="year",
    cat_var="carrier",
    ylim=None,
    figsize=(4, 4),
    make_fig=True,
):
    if carrier_name is None:
        carrier_name = (
            bus_carriers if isinstance(bus_carriers, str) else bus_carriers[0]
        )

    if network is not None:

        df = calculate_storage_capacities(
            network, bus_carriers=bus_carriers, bus_qualifiers=bus_qualifiers
        )

        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = (df["value"] / 1e3).round(2)

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )

    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title=f"Installed {carrier_name} storage capacity [GWh]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 0.5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, df


def plot_storage_capacity_additions(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=["battery large electricity", "hydro PSH electricity"],
    carrier_name=None,
    bus_qualifiers=None,
    x_var="year",
    cat_var="carrier",
    ylim=None,
    figsize=(5, 4),
    make_fig=True,
):
    if carrier_name is None:
        carrier_name = (
            bus_carriers if isinstance(bus_carriers, str) else bus_carriers[0]
        )

    if network is not None:

        df = calculate_storage_capacity_additions(
            network, bus_carriers=bus_carriers, bus_qualifiers=bus_qualifiers
        )
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = (df["value"] / 1e3).round(2)

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title=f"{carrier_name.capitalize()} storage capacity additions [GWh]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 1.5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_grid_capacities(
    network=None,
    df=None,
    run_name=None,
    x_var="year",
    cat_var="carrier",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
):
    if network is not None:
        df = calculate_grid_capacities(network)
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = (df["value"] / 1e3).round(2)

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title="Grid capacities [GW]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, df


def plot_grid_capacity_additions(
    network=None,
    df=None,
    run_name=None,
    x_var="year",
    cat_var="carrier",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
):
    if network is not None:
        df = calculate_grid_capacity_additions(network)
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = (df["value"] / 1e3).round(2)

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title="Grid capacity additions [GW]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_annual_generation(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=["electricity in", "electricity out"],
    carrier_name="electricity",
    bus_qualifiers=None,
    x_var="year",
    cat_var="carrier",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
):
    if carrier_name is None:
        carrier_name = (
            bus_carriers if isinstance(bus_carriers, str) else bus_carriers[0]
        )

    if network is not None:

        df = calculate_flows(network, bus_carriers, bus_qualifiers=bus_qualifiers)
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = df["value"].round(2)
        # df = df[~df[cat_var].str.contains("final use")]

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title=f"{carrier_name.capitalize()} generation [TWh]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )

    return fig, df


def plot_energy_balance_at_peak_load(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=["electricity in", "electricity out"],
    carrier_name=None,
    bus_qualifiers=None,
    x_var="year",
    cat_var="carrier",
    load_type="total",
    ylim=None,
    figsize=(5, 8),
    make_fig=True,
):

    if carrier_name is None:
        carrier_name = (
            bus_carriers if isinstance(bus_carriers, str) else bus_carriers[0]
        )

    if network is not None:
        df = calculate_energy_balance_at_peak_load(
            network,
            bus_carriers=bus_carriers,
            bus_qualifiers=bus_qualifiers,
            cat_var=cat_var,
            year_share=40 / 8760,
            load_type=load_type,
        )
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = df["value"].round(2)

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title=f"{carrier_name.capitalize()} generation at peak {load_type} load hours [GW]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )

    return fig, df


def plot_curtailed_vres_energy(
    network=None,
    df=None,
    run_name=None,
    x_var="year",
    cat_var="carrier",
    ylim=None,
    figsize=(5, 4),
    make_fig=True,
):

    if network is not None:
        df = calculate_curtailed_vres_energy(network)
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = (df["value"] / 1e6).round(2)

    if df.empty:
        return plt.figure(), df

    carrier_order, carrier_colors = get_order_and_colors(
        network, agg=cat_var, run_name=run_name
    )
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title="Curtailed vRES energy [TWh]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 0.5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_fuel_consumption(
    network=None,
    df=None,
    run_name=None,
    x_var="year",
    cat_var="fuel",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
):

    if network is not None:
        df = calculate_fuel_consumption(network, unit="PJ")
        df = df.groupby(["year", cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = df["value"].round(1)
        # Exclude process emissions and lulucf
        df = df[~df[cat_var].isin(["process emissions", "lulucf"])]

    carrier_order, carrier_colors = get_order_and_colors(network, run_name=run_name)
    if cat_var == "fuel":
        intermediate_fuels = {
            "biogas upgrading": "biomethane supply",
            "biogas production": "biogas supply",
        }
        for old, new in intermediate_fuels.items():
            if old in carrier_colors:
                carrier_order = carrier_order.str.replace(old, new)
                carrier_colors[new] = carrier_colors.pop(old)
        carrier_order = carrier_order[carrier_order.str.endswith(" supply")].str[
            : -len(" supply")
        ]
        carrier_colors = {
            fuel: carrier_colors[f"{fuel} supply"] for fuel in carrier_order
        }
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title="Fuel consumption [PJ]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 30),
        label_digits=0,
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, df


def plot_co2_emissions(
    network=None,
    df=None,
    run_name=None,
    x_var="year",
    cat_var="fuel",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
):

    if network is not None:
        df = calculate_co2_emissions(network)
        df = df.groupby(["year", cat_var]).agg({"value": "sum"}).reset_index()
        df["value"] = df["value"].round(2)

    carrier_order, carrier_colors = get_order_and_colors(network, run_name=run_name)
    if cat_var == "fuel":
        extra_carriers = {
            "biogas upgrading": "biogas upgrading supply",
            "biomass agriculture CHP CC": "biomass agriculture CHP CC supply",
        }
        for old, new in extra_carriers.items():
            if old in carrier_colors:
                carrier_order = carrier_order.str.replace(old, new)
                carrier_colors[new] = carrier_colors.pop(old)
        carrier_order = carrier_order[carrier_order.str.endswith(" supply")].str[
            : -len(" supply")
        ]
        carrier_colors = {
            fuel: carrier_colors[f"{fuel} supply"] for fuel in carrier_order
        }
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title="CO₂ emissions [Mt]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, df


def plot_opex(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=None,
    cost_attr="marginal_cost",
    x_var="year",
    cat_var="aggregation",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
    area="PL",
):
    title = {
        "marginal_cost": "Total variable costs [bln PLN]",
        "variable_cost": "Variable O&M costs [bln PLN]",
        "co2_cost": "CO₂ costs [bln PLN]",
    }

    if network is not None:

        df = calculate_opex(network, cost_attr=cost_attr, bus_carriers=bus_carriers)
        # Only include domestic costs, without electricity trade balance
        df = df[df["area"].str.startswith(area)]
        df = df.groupby(["year", cat_var]).agg({"value": "sum"}).reset_index()
        # Convert to bln PLN
        df["value"] = (df["value"] / 1e9).round(2)
        df = df[df["value"].abs() > 0]

    order, colors = get_order_and_colors(network, agg=cat_var, run_name=run_name)
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title=title[cost_attr],
        x_var=x_var,
        cat_var=cat_var,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_capex(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=None,
    cost_attr="capital_cost",
    x_var="year",
    cat_var="aggregation",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
    area="PL",
):
    title = {
        "capital_cost": "Total capital costs [bln PLN]",
        "fixed_cost": "Fixed O&M costs [bln PLN]",
        "annual_investment_cost": "Annual investment costs [bln PLN]",
        "investment_cost": "Overnight investment costs [bln PLN]",
    }

    if network is not None:

        df = calculate_capex(network, cost_attr=cost_attr, bus_carriers=bus_carriers)
        # Include only domestic costs
        df = df[df["area"].str.startswith(area)]
        df = df.groupby([x_var, cat_var]).agg({"value": "sum"}).reset_index()
        # Convert to bln PLN
        df["value"] = (df["value"] / 1e9).round(2)
        df = df[df["value"].abs() > 0]

    if df.empty:
        return plt.figure(), df

    order, colors = get_order_and_colors(network, agg=cat_var, run_name=run_name)
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title=title[cost_attr],
        x_var=x_var,
        cat_var=cat_var,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 2),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_detailed_costs(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=None,
    cat_var="aggregation",
    ylim=None,
    figsize=(6, 6),
    make_fig=True,
    area="PL",
):

    costs = [
        ("variable_cost", "Var. O&M", calculate_opex),
        ("co2_cost", "CO₂", calculate_opex),
        ("fixed_cost", "Fix. O&M", calculate_capex),
        ("annual_investment_cost", "Ann. invest.", calculate_capex),
    ]

    if network is not None:

        dfs = []
        for cost_attr, label, calculate_cost in costs:
            df = calculate_cost(network, cost_attr=cost_attr, bus_carriers=bus_carriers)
            # Only include domestic costs, without electricity trade balance
            df = df[df["area"].str.startswith(area)]
            df = df.groupby(["year", cat_var]).agg({"value": "sum"}).reset_index()
            df["cost component"] = label
            dfs.append(df)
        df = pd.concat(dfs)

        df = df[df["year"] == network.meta["year"]].drop(columns="year")
        # Convert to bln PLN
        df["value"] = (df["value"] / 1e9).round(4)
        df = df[df["value"].abs() > 0]

    df["cost component"] = pd.Categorical(
        df["cost component"], categories=[label for _, label, _ in costs], ordered=True
    )

    order, colors = get_order_and_colors(network, agg=cat_var, run_name=run_name)
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        x_var="cost component",
        title="Annual cost components [bln PLN]",
        cat_var=cat_var,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel(network.meta["year"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_average_unit_cost_and_price(
    network=None, df=None, run_name=None, ylim=None, figsize=(4, 4), make_fig=True
):
    # TODO: generalise for a case with separate final use carriers

    metrics = ["Avg. unit cost", "Avg. price"]

    if network is not None:

        df = calculate_statistics(network)
        df = df[
            [
                "year",
                "carrier",
                "Withdrawal",
                "Revenue",
                "Operational Expenditure",
                "Capital Expenditure",
            ]
        ]
        df["Total Cost"] = df["Operational Expenditure"] + df["Capital Expenditure"]

        df_cost = (
            df.groupby(["year"])
            .agg(**{"Avg. unit cost": ("Total Cost", "sum")})
            .reset_index()
        )
        is_final_use = df["carrier"].str.contains("final use")
        df_price_use = (
            df[is_final_use]
            .groupby(["year"])
            .agg(
                **{
                    "Avg. price": ("Revenue", lambda x: -np.sum(x)),
                    "Final use": ("Withdrawal", "sum"),
                }
            )
            .reset_index()
        )

        df = pd.merge(df_cost, df_price_use, on="year")
        for col in metrics:
            df[col] = (df[col] / df["Final use"]).round(1)
        df = df.melt(
            id_vars="year", value_vars=metrics, var_name="metric", value_name="value"
        )
        df = df[df["year"] == network.meta["year"]].drop(columns="year")

    df["metric"] = pd.Categorical(df["metric"], categories=metrics, ordered=True)
    df["metric2"] = df["metric"]

    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        x_var="metric",
        cat_var="metric2",
        title="Average unit cost and price [PLN/MWh]",
        cat_colors={metric: "#535ce3" for metric in metrics},
        show_total_label=False,
        figsize=figsize,
    )

    df = df.drop(columns="metric2")

    ax = fig.axes[0]
    ax.set_xlabel(network.meta["year"])
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Remove legend
    ax.get_legend().remove()

    return fig, df


def plot_total_costs(
    network=None,
    df=None,
    run_name=None,
    costs=["OPEX", "CAPEX"],
    bus_carriers=None,
    x_var="year",
    cat_var="aggregation",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
    area="PL",
):

    if network is not None:
        df = calculate_statistics(network, bus_carriers=bus_carriers)
        # Only include domestic costs, without electricity trade balance
        df = df[df["area"].str.startswith(area)]
        df = df[[x_var, cat_var, "Operational Expenditure", "Capital Expenditure"]]
        df["Total costs"] = 0
        if "OPEX" in costs:
            df["Total costs"] += df["Operational Expenditure"]
        if "CAPEX" in costs:
            df["Total costs"] += df["Capital Expenditure"]

        df = (
            df.groupby([x_var, cat_var]).agg(value=("Total costs", "sum")).reset_index()
        )
        # Convert to bln PLN
        df["value"] = (df["value"] / 1e9).round(4)
        df = df[df["value"].abs() > 0]
        # df = df[df["carrier"] != "electricity final use"]

    order, colors = get_order_and_colors(network, agg=cat_var, run_name=run_name)
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title="Total annual costs [bln PLN]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 2),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_net_revenues(
    network=None,
    df=None,
    run_name=None,
    costs=["OPEX", "CAPEX"],
    x_var="year",
    cat_var="aggregation",
    ylim=None,
    figsize=(4, 6),
    make_fig=True,
    area="PL",
):

    if network is not None:
        df = calculate_statistics(network)
        # Only include domestic costs, without electricity trade balance
        df = df[df["area"].str.startswith(area)]
        df = df[
            [
                x_var,
                cat_var,
                "Revenue",
                "Operational Expenditure",
                "Capital Expenditure",
            ]
        ]
        df["Net revenue"] = df["Revenue"]
        if "OPEX" in costs:
            df["Net revenue"] -= df["Operational Expenditure"]
        if "CAPEX" in costs:
            df["Net revenue"] -= df["Capital Expenditure"]
        df = (
            df.groupby([x_var, cat_var]).agg(value=("Net revenue", "sum")).reset_index()
        )
        # Convert to bln PLN
        df["value"] = (df["value"] / 1e9).round(2)
        df = df[df["value"].abs() > 0]
        df = df[df[cat_var] != "electricity final use"]

    order, colors = get_order_and_colors(network, agg=cat_var)
    if not make_fig:
        return None, df

    fig = plot_bar(
        df,
        title="Net revenue [bln PLN]",
        x_var=x_var,
        cat_var=cat_var,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


# Custom datasets


def format_output_table(df, x_var, y_var, cat_var, cat_order, show_total_label, digits):

    df = df.pivot(index=cat_var, columns=x_var, values=y_var).fillna(0)
    df.columns = df.columns.astype(str)
    df = df.loc[cat_order.intersection(df.index), ~df.columns.isin(["", " ", "\n"])]
    df.index = (
        df.index.str.replace("\n", " ").str.replace("  ", " ").str.replace("  ", " ")
    )
    df.columns = (
        df.columns.str.replace("\n", " ").str.replace("  ", " ").str.replace("  ", " ")
    )
    df.columns.name = f"{cat_var} / {x_var}"
    if show_total_label:
        df.loc["TOTAL"] = df.sum().round(digits)
    return df


def plot_custom_dataset_as_barplot(
    df=None,
    run_name=None,
    y_var="value",
    x_var="year",
    cat_var="carrier",
    title="Custom plot [unit]",
    custom_order_and_colors=None,
    digits=2,
    show_total_label=True,
    label_commas=False,
    ylim=None,
    unit=None,
    figsize=(5, 6),
    make_fig=True,
):

    df = df.groupby([x_var, cat_var], observed=True).agg({y_var: "sum"}).reset_index()
    df["value"] = df["value"].round(digits)
    df = df[df["value"].abs() > 0]

    if df.empty:
        return plt.figure(), df

    if custom_order_and_colors is not None:
        order = pd.Index(custom_order_and_colors.keys())
        colors = custom_order_and_colors  # dict
    else:
        order, colors = get_order_and_colors(None, agg=cat_var, run_name=run_name)

    missing_categories = set(df[cat_var]).difference(order)
    if len(missing_categories) > 0:
        logging.warning(f"Missing categories from order: {missing_categories}")

    df_output = format_output_table(
        df,
        x_var,
        y_var,
        cat_var,
        order,
        show_total_label,
        digits,
    )

    if not make_fig:
        return None, df_output

    fig = plot_bar(
        df,
        title=title,
        x_var=x_var,
        cat_var=cat_var,
        cat_order=order,
        cat_colors=colors,
        label_digits=digits - 1,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        label_commas=label_commas,
        show_total_label=show_total_label,
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if unit is None:
        unit = title.split("[")[1][:-1]
    ax.set_ylabel(unit, ha="left", y=1, rotation=0, labelpad=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
        if ylim[0] * ylim[1] < 0:
            ax.axhline(
                y=0,
                linestyle="--",
                color=mpl_style["axes.edgecolor"],
                linewidth=mpl_style["axes.linewidth"],
            )

    return fig, df_output


# Hourly quantities


def plot_hourly_generation(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=["electricity in", "electricity out"],
    carrier_name="electricity",
    bus_qualifiers=None,
    subperiods=None,
    cat_var="carrier",
    ylim=None,
    unit="GW",
    figsize=(8, 6),
    custom_order_and_colors=None,
    make_fig=True,
):

    if carrier_name is None:
        carrier_name = (
            bus_carriers if isinstance(bus_carriers, str) else bus_carriers[0]
        )

    if network is not None:

        df = calculate_flows(
            network, bus_carriers, bus_qualifiers=bus_qualifiers, annual=False
        )
        df = df.groupby(level=[cat_var]).sum()
        df = (df / 1e3).round(3)
        df = df.transpose()
        df = df.drop(
            columns=(
                []
                # [col for col in df.columns if "final use" in col]
                # + [col for col in df.columns if col.endswith(("storage", "battery"))]
            )
        )

    df.index = pd.to_datetime(df.index)
    if not make_fig:
        return None, df

    if subperiods is None:
        subperiods = [("", (0, len(df)))]

    if custom_order_and_colors is None:
        carrier_order, carrier_colors = get_order_and_colors(
            network, agg=cat_var, run_name=run_name
        )
    else:
        carrier_order = pd.Index(custom_order_and_colors.keys())
        carrier_colors = custom_order_and_colors

    for subperiod, (i_start, i_stop) in subperiods:
        subdf = df.iloc[i_start:i_stop, :]

        fig = plot_area(
            subdf,
            is_wide=True,
            title=f"Hourly {carrier_name} generation{' – ' if subperiod != '' else ''}{subperiod} [GW]",
            cat_var=cat_var,
            cat_order=carrier_order,
            cat_colors=carrier_colors,
            ylim=ylim,
            figsize=figsize,
        )

        ax = fig.axes[0]
        ax.set_xlabel("")
        # if ylim is not None:
        #     ax.set_ylim(*ylim)
        ax.axhline(
            y=0,
            linestyle="--",
            color=mpl_style["axes.edgecolor"],
            linewidth=mpl_style["axes.linewidth"],
        )

        ax.set_ylabel(unit, ha="left", y=1, rotation=0, labelpad=0)

        ticks = subdf.index[subdf.index.hour == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks.strftime("%d.%m\n%H:%M"))
        ax.set_xlabel(None)

    # Returns fig for the last subperiod
    return fig, df


def plot_prices(
    network=None,
    df=None,
    run_name=None,
    bus_carriers=["electricity in", "electricity out"],
    carrier_name="electricity",
    bus_qualifiers=None,
    subperiods=None,
    ylim=None,
    figsize=(8, 4),
    make_fig=True,
):

    if carrier_name is None:
        carrier_name = (
            bus_carriers if isinstance(bus_carriers, str) else bus_carriers[0]
        )

    if network is not None:
        df = calculate_marginal_prices(
            network, bus_carriers, bus_qualifiers=bus_qualifiers
        ).transpose()

    df.index = pd.to_datetime(df.index)

    if subperiods is None:
        subperiods = [("", (0, len(df)))]

    for subperiod, (i_start, i_stop) in subperiods:
        subdf = df.iloc[i_start:i_stop, :]

        fig = plot_line(
            subdf,
            title=f"Marginal {carrier_name} cost{' – ' if subperiod != '' else ''}{subperiod} [PLN/MWh]",
            figsize=figsize,
        )

        ax = fig.axes[0]
        ax.set_xlabel("")
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.legend().remove()
        ticks = subdf.index[subdf.index.hour == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks.strftime("%d.%m\n%H:%M"))
        ax.set_xlabel(None)

    # Returns fig for the last subperiod
    return fig, df


generation_carrier_mapping = {
    "wind offshore": "wind offshore",
    "wind onshore": "wind onshore",
    "solar PV roof": "solar PV roof",
    "solar PV ground": "solar PV roof",
    "nuclear power": "nuclear power",
    "_other": "natural gas power",  # map all others into color of natural gas power
}
generation_labels = {
    "wind offshore": "wind offshore",
    "wind onshore": "wind onshore",
    "solar PV roof": "solar PV",
    "nuclear power": "nuclear",
    "natural gas power": "other generation",
}


consumption_carrier_mapping = {
    "hydrogen electrolysis": "hydrogen electrolysis",
    "electricity HMV final use": "electricity HMV final use",
    "electricity LV final use": "electricity HMV final use",
    "_other": "electricity final use",  # map all others into color of electricity final use
}
consumption_labels = {
    "hydrogen electrolysis": "electrolysis",
    "electricity HMV final use": "other electricity use",
}

translations_pl = {
    "wind offshore": "wiatr – morze",
    "wind onshore": "wiatr – ląd",
    "solar PV": "PV",
    "nuclear": "energia jądrowa",
    "other generation": "pozostałe źródła",
    "electrolysis": "elektroliza",
    "other electricity use": "pozostałe zużycie energii",
    "Consumption": "Pobór",
    "Generation": "Produkcja",
    "Transmission capacity": "Moce przesyłowe",
    "Mean line load": "Średnie obciążenie linii",
}

t = lambda x: translations_pl.get(x, x)

cm_to_inches = lambda x, y: (x / 2.54, y / 2.54)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_power_system(network, transmission_agg="mean"):

    projection = ccrs.TransverseMercator(central_longitude=20, central_latitude=52)
    ax = plt.axes(projection=projection)
    fig = ax.figure
    fig.set_size_inches(cm_to_inches(18, 15))

    gdf = gpd.read_file(data_dir("clean", "voivodeships.geojson"))
    bounds = gdf.total_bounds
    margin = 0.1  # 0.1
    bounds = [
        bounds[0] - margin,
        bounds[2] + margin,
        bounds[1] - margin,
        bounds[3] + margin,
    ]

    ax.add_geometries(
        gdf.geometry,
        facecolor="none",
        edgecolor="black",
        lw=0.5,
        crs=ccrs.PlateCarree(),
    )

    df_xy = pd.read_csv(data_dir("clean", "voivodeship_centroids.csv"))
    df_xy["area"] = "PL " + df_xy["voivodeship"]
    df_xy = pd.concat(
        [
            df_xy,
            pd.read_csv(data_dir("clean", "neighbour_centroids.csv")),
        ],
        ignore_index=True,
    )
    df_xy = df_xy.set_index("area")[["lon", "lat"]]

    network.buses.x = network.buses.area.map(df_xy["lon"])
    network.buses.y = network.buses.area.map(df_xy["lat"])

    alpha = 1
    gw_to_width = 1.3

    # BUSES: show positive (production) and negative (consumption) power flows
    flows = (
        network.statistics.energy_balance(
            groupby=["bus", "area", "carrier"],
            bus_carrier=["electricity in", "electricity out"],
            aggregate_time="sum",
        )
    ).reset_index()
    flows = flows[["bus", "area", "carrier", network.meta["year"]]].rename(
        columns={network.meta["year"]: "value"}
    )
    # Remove foreign buses
    flows = flows[flows["area"].str.startswith("PL ")]
    # Remove interarea flows
    flows = flows[
        ~flows["carrier"].isin(
            [
                "transmission line AC",
                "transmission line DC",
                "electricity import",
                "electricity export",
            ]
        )
    ]
    # Remove storage and grid losses
    flows = flows[
        ~flows["carrier"].str.startswith(("battery", "hydro PSH", "direct line"))
    ]
    flows = flows[
        ~flows["carrier"].str.startswith(
            ("distribution", "transformation", "connection")
        )
    ]

    # Remap HMV and LV flows to EHV buses
    flows["bus"] = flows["area"] + " electricity in EHV"

    flows_agg = flows.groupby(["carrier"])["value"].sum().reset_index()
    generation_carriers = flows_agg[flows_agg["value"] > 0]["carrier"].unique()
    consumption_carriers = flows_agg[flows_agg["value"] < 0]["carrier"].unique()

    # Use custom groupby
    flows.loc[flows["carrier"].isin(generation_carriers), "carrier"] = (
        flows.loc[flows["carrier"].isin(generation_carriers), "carrier"]
        .map(generation_carrier_mapping)
        .fillna(generation_carrier_mapping["_other"])
    )

    flows.loc[flows["carrier"].isin(consumption_carriers), "carrier"] = (
        flows.loc[flows["carrier"].isin(consumption_carriers), "carrier"]
        .map(consumption_carrier_mapping)
        .fillna(consumption_carrier_mapping["_other"])
    )

    # Get unique carriers and their colors for legend
    unique_carriers = flows["carrier"].unique()
    carrier_colors = {}
    for carrier in unique_carriers:
        if carrier in network.carriers.index:
            carrier_colors[carrier] = network.carriers.loc[carrier, "color"]
        else:
            # Fallback color if not in network.carriers
            carrier_colors[carrier] = "gray"

    # Reorder elements
    flows["order"] = flows["carrier"].map(network.carriers["order"])
    flows = flows.groupby(["bus", "carrier", "order"]).sum().reset_index()
    flows = flows.sort_values(["bus", "order"]).drop(columns="order")
    flows = flows.set_index(["bus", "carrier"])["value"]

    ordered_carriers = sorted(
        unique_carriers, key=lambda x: network.carriers.loc[x, "order"]
    )

    # LINES: width shows transmission volume, color shows annual utilisation

    # Aggregate parallel lines, name according to first line
    buses_to_representative_line = (
        network.lines.reset_index(names="name")
        .groupby(["bus0", "bus1"])["name"]
        .first()
    )
    line_to_representative_line = network.lines[["bus0", "bus1"]].apply(
        lambda row: buses_to_representative_line.loc[(row["bus0"], row["bus1"])], axis=1
    )

    transmission_capacity = network.lines.groupby(line_to_representative_line)[
        "s_nom_opt"
    ].sum()

    # Sum p0 in groups as defined by line_to_representative_line
    p0_agg = network.lines_t.p0.T.groupby(line_to_representative_line).sum().T

    if transmission_agg == "mean":
        transmission_direction = p0_agg.mean().apply(np.sign) * transmission_capacity
        transmission_utilisation = p0_agg.abs().mean() / transmission_capacity * 100
    # elif transmission_agg == "max":
    #     idxmax = network.lines_t.p0.abs().idxmax()
    #     transmission_direction = (
    #         pd.Series(
    #             [network.lines_t.p0.loc[t, i] for i, t in idxmax.items()],
    #             index=idxmax.index,
    #         ).apply(np.sign)
    #         * transmission_capacity
    #     )
    #     transmission_utilisation = (
    #         pd.Series(
    #             [network.lines_t.p0.loc[t, i] for i, t in idxmax.items()],
    #             index=idxmax.index,
    #         ).abs()
    #         / transmission_capacity
    #     )

    transmission_direction.index = pd.MultiIndex.from_product(
        [["Line"], transmission_direction.index], names=["component", "name"]
    )
    transmission_utilisation_max = transmission_utilisation.max()
    transmission_utilisation_max = 0.65 * 100

    # LINKS: interconnectors
    # Group by the neighbour country, and calculate mean capacity over directions
    interconnectors = network.links[
        network.links.carrier.isin(["electricity import", "electricity export"])
    ].copy()
    interconnectors["neighbour_area"] = np.where(
        interconnectors.area.str.startswith("PL"),
        interconnectors.area_from,
        interconnectors.area,
    )
    interconnectors["domestic_area"] = np.where(
        interconnectors.area.str.startswith("PL"),
        interconnectors.area,
        interconnectors.area_from,
    )

    interconnector_capacity = (
        interconnectors.reset_index(names="name")
        .groupby(["neighbour_area", "domestic_area"])
        .agg({"p_nom_opt": "sum", "name": "first"})
        .set_index("name")["p_nom_opt"]
    )

    # Divide by 2 to get mean capacity over both directions
    interconnector_capacity /= 2

    collections = network.plot(
        ax=ax,
        title="",# f"{network.name} - {network.meta['year']}",
        bus_sizes=flows / flows.abs().sum() * 1.2,
        bus_split_circles=True,
        # bus_colors=network.buses.color,
        line_widths=transmission_capacity / 1000 * gw_to_width,
        link_widths=interconnector_capacity / 1000 * gw_to_width,
        line_colors=transmission_utilisation,
        link_colors="grey",
        line_cmap="YlOrRd",
        line_norm=Normalize(
            vmin=0, vmax=ceil(transmission_utilisation_max * 100) / 100
        ),
        line_alpha=alpha,
        link_alpha=alpha,
        # flow=transmission_direction / 1000 * gw_to_width * 0.75,
        projection=projection,
        geomap=True,
        color_geomap=False,
        boundaries=bounds,
    )

    # legend_labels = [
    #     "  AC 400 kV 2 GVA",
    #     "  AC 400 kV 4 GVA",
    #     "  AC 400 kV 6 GVA",
    # ]

    # pypsa.plot.add_legend_lines(
    #     ax,
    #     sizes=[2 * gw_to_width, 4 * gw_to_width, 6 * gw_to_width],
    #     labels=legend_labels,
    #     colors=["orange"] * len(legend_labels),
    #     patch_kw=None,
    #     legend_kw=dict(title=t("Transmission capacity")),
    # )

    transmission_legend_elements = [
        Line2D([0], [0], color="orange", lw=2 * gw_to_width, label="  2 GVA"),
        Line2D([0], [0], color="orange", lw=4 * gw_to_width, label="  4 GVA"),
        Line2D([0], [0], color="orange", lw=6 * gw_to_width, label="  6 GVA"),
    ]

    # Create the transmission capacity legend
    transmission_legend = ax.legend(
        handles=transmission_legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.10, 1),
        title=t("Transmission capacity"),
        title_fontsize=9,
        # fontsize=10
    )

    # Add the transmission legend to the plot
    ax.add_artist(transmission_legend)

    # Add carrier legend (generation and consumption)
    legend_elements = []

    generation_carriers_sorted = [
        c for c in reversed(ordered_carriers) if c in generation_carriers
    ]
    consumption_carriers_sorted = [
        c for c in reversed(ordered_carriers) if c in consumption_carriers
    ]

    # Add generation carriers
    if generation_carriers_sorted:
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label=f"{t('Generation')} [TWh]")
        )
        for carrier in generation_carriers_sorted:
            legend_elements.append(
                Patch(
                    facecolor=carrier_colors[carrier],
                    label=f"  {t(generation_labels[carrier])}",
                )
            )

    # Add consumption carriers
    if consumption_carriers_sorted:
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label=f"{t('Consumption')} [TWh]")
        )
        for carrier in consumption_carriers_sorted:
            legend_elements.append(
                Patch(
                    facecolor=carrier_colors[carrier],
                    label=f"  {t(consumption_labels[carrier])}",
                )
            )

    # Create the carriers legend
    carriers_legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0, 0),
        frameon=False,
        title="",
    )

    # Add the carriers legend to the plot
    ax.add_artist(carriers_legend)

    colorbar = next(x for x in collections if x.get_label() == "_child2")
    fig.colorbar(
        colorbar,
        fraction=0.02,
        pad=0.004,
        label=f"{t(transmission_agg.capitalize() + ' line load')} [%]",
    )

    fig.tight_layout()
    # legend_handles = [
    #     Line2D([0], [0], color="red", lw=1 * gw_to_width, alpha=alpha),
    #     Line2D([0], [0], color="red", lw=2 * gw_to_width, alpha=alpha),
    # ]
    # ax.legend(
    #     handles=legend_handles,
    #     labels=legend_labels,
    #     loc="center left",
    #     bbox_to_anchor=(1, 0.5),
    #     frameon=False,
    # )

    # ax.figure.tight_layout()

    # name = f"grid_pl{'_reduced' if reduced else ''};v_noms={'+'.join(str(v) for v in v_noms)}"
    # ax.figure.savefig(
    #     data_dir("clean", "electricity_grid_data", f"{name}.png"),
    #     bbox_inches="tight",
    #     dpi=600,
    # )
    plt.close()
    return fig
