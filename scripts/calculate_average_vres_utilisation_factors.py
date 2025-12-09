import pandas as pd
import os

from pypsa_pl.config import data_dir

# This script calculates the average availability factors for variable renewable energy sources based on the raw timeseries data
# User can modify the availability profiles by increasing or decreasing the annual average factors in the input technology data
# According to the analysis of monthly ARE data, the average availability factors for vRES in Poland in recent years are:
# - wind onshore (all): 29% (all turbines)
# --> according to DAE data post-2020 turbines had ~10% higher availability factor than pre-2020 turbines, hence estimate
# - wind onshore old: 28% (pre-2020 turbines, around 70% of total capacity)
# - wind onshore: 31% (post-2020 turbines. around 30% of total capacity)
# --> 0.7 * 0.28 + 0.3 * 0.31 = ~0.29 
# - solar PV ground: 11.5% (professional PV units)
# - solar PV roof: 10.5% (prosumer PV units, assuming ~25% self-consumption)
# We consider 2012 to be a typical weather year, so the availability corrections are calibrated using the 2012 average factors


def calculate_average_vres_availability_factors_raw():

    os.makedirs(data_dir("clean", "average_vres_availability_factors"), exist_ok=True)

    for weather_year in [2012, 2013, 2015]:
        dfs = []

        for technology in [
            "wind onshore",
            "wind onshore old",
            "wind offshore",
            "solar PV ground",
            "solar PV ground E",
            "solar PV ground W",
            "solar PV roof",
        ]:
            df = pd.read_csv(
                data_dir(
                    "input",
                    "timeseries;variant=full",
                    f"availability_profile;technology={technology};year={weather_year}.csv",
                ),
                index_col=0,
            )
            df = df.mean(axis=0).round(3).to_frame(name=technology).T
            dfs.append(df)

        df = pd.concat(dfs).reset_index()
        df.to_csv(
            data_dir(
                "clean",
                "average_vres_availability_factors",
                f"average_vres_availability_factors_raw;weather_year={weather_year}",
            ),
            index=False,
        )


if __name__ == "__main__":

    calculate_average_vres_availability_factors_raw()
