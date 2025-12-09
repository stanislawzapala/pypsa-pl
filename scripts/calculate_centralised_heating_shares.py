import pandas as pd
import os

from pypsa_pl.config import data_dir

# PyPSA-PL assumes that the ratio of centralised to decentralised heating in each voivodeship is fixed according to observed trends
# Estimation of this voivodeship-specific ratio is however model-dependent
# We estimate the baseline shares based on the results of the 2025 model run


def calculate_centralised_heating_shares():
    os.makedirs(data_dir("clean", "centralised_heating_shares"), exist_ok=True)

    # (1) Read the 2025 installed capacity data to establish the baseline centralised heating shares
    # The centralised heating share does not depend on the specific variant, as 2025 is universal for all variants
    df = pd.read_csv(
        data_dir(
            "input",
            "installed_capacity;variant=instrat_ambitious+trade+micro.csv",
        ),
        index_col=0,
    )
    # To get also the collective result for all voivodeships, we duplicate all the data and assign the "PL" area
    df = pd.concat(
        [
            df,
            df.assign(area="PL"),
        ],
        ignore_index=True,
    )

    is_2025 = df["build_year"] == 2025
    is_centralised = df["technology"].isin(
        [
            "centralised space heating",
            "centralised water heating",
            "centralised other heating",
        ]
    )
    is_decentralised = df["technology"].isin(
        [
            "decentralised space heating",
            "decentralised water heating",
        ]
    )
    # Addition of building retrofits capacity in 2025 is the total building retrofits capacity by definition
    # So we do not need to worry that we will miss any preexisting capacity
    is_centralised |= (df["technology"] == "building retrofits") & (
        df["qualifier"] == "heat centralised"
    )
    is_decentralised |= (df["technology"] == "building retrofits") & (
        df["qualifier"] == "heat decentralised"
    )

    dfs = {
        "centralised": df.loc[
            is_2025 & is_centralised, ["area", "technology", "nom"]
        ].copy(),
        "decentralised": df.loc[
            is_2025 & is_decentralised, ["area", "technology", "nom"]
        ].copy(),
    }

    # Convert capacities into flows using the PyPSA-PL default capacity utilisation factors
    # One needs to verify whether these factors are still valid!
    params = {
        "space_heating_utilisation": 0.1,
        "water_heating_utilisation": 1,
        "other_heating_utilisation": 1,
        "heat_capacity_utilisation": 0.2,
    }

    utilisation_factors = {
        "decentralised space heating": params["space_heating_utilisation"],
        "decentralised water heating": params["water_heating_utilisation"],
        "centralised space heating": params["space_heating_utilisation"],
        "centralised water heating": params["water_heating_utilisation"],
        "centralised other heating": params["other_heating_utilisation"],
        "building retrofits": params["heat_capacity_utilisation"],
    }
    for key, df in dfs.items():
        df["flow"] = df["nom"] * df["technology"].map(utilisation_factors)
        dfs[key] = df.groupby("area")["flow"].sum()

    # Calculate the centralised heating share
    df = 1 / (1 + dfs["decentralised"] / dfs["centralised"])

    # Indcate that the results are for the year 2025
    df = df.to_frame(name=2025)

    # (2) Combine with the PL-wide long term projection to get voivodeship specific shares
    # The following functional form is assumed
    # r(a, y) = centralised heating share in area a in year y
    # r(a, y) = 1 / [1 + (1 / r(a, 2025) - 1) * (1 / r(PL, y) - 1) / (1 / r(PL, 2025) - 1)]

    # Load the file with PL-wide share projection
    # This file has to be imported from energy-modelling repository
    df_pl = pd.read_csv(
        data_dir(
            "clean",
            "centralised_heating_shares",
            "assumed_heat_centralised_and_decentralised_shares.csv",
        ),
    )
    df_pl = df_pl.set_index("year")["heat_centralised_share"]
    for year in df_pl.index:
        if year == 2025:
            continue
        # Calculate the centralised heating share for each voivodeship
        df[year] = 1 / (
            1 + (1 / df[2025] - 1) * (1 / df_pl[year] - 1) / (1 / df_pl[2025] - 1)
        )
    df = df.round(3)

    # Reorder the columns to have the years in ascending order
    df = df.reindex(sorted(df.columns), axis=1)

    # Save the centralised heating shares for each voivodeship
    df.to_csv(
        data_dir(
            "clean", "centralised_heating_shares", "centralised_heating_shares.csv"
        )
    )


if __name__ == "__main__":
    calculate_centralised_heating_shares()
