# PyPSA-PL: optimisation model of the Polish energy system


## Introduction

PyPSA-PL is an implementation of the energy modelling framework [PyPSA](https://pypsa.readthedocs.io) shipped with a use-ready dataset tailored for the Polish energy system. PyPSA-PL can be used to plan optimal investments in the power, heating, hydrogen, and light vehicle sectors – given final-use demand together with capital and operational costs for assets – or to optimise the hourly dispatch of utility units – given final-use demand and operational costs only. Starting with v3.1 (released in 2025), the model can operate on 16 voivodeship-level nodes and accounts for transmission and distribution expansion costs. All that makes it a valuable tool for investigating the feasibility of decarbonisation scenarios for the Polish energy system, in which a large share of electricity is supplied by variable sources such as wind and solar.

![](docs/pypsa_pl.png)


## Installation and usage

PyPSA-PL has been developed and tested using Python 3.10. The project dependencies can be installed using the [Poetry](https://python-poetry.org/) tool according to the [pyproject.toml](pyproject.toml) file. Alternatively, you can use any other Python package manager – the dependencies are also listed in the [requirements.txt](requirements.txt) file. Additionally, you will need to install an external solver (see [PyPSA manual](https://pypsa.readthedocs.io/en/latest/installation.html#getting-a-solver-for-optimisation)). 

PyPSA-PL-mini notebooks can be deployed on the [Google Colab](https://colab.google/) platform. To do so, navigate to one of the PyPSA-PL-mini application notebooks in the [notebooks](notebooks) directory. In the notebook, click the "Open in Colab" banner and follow the instructions provided therein.


## Input data and assumptions

This table lists the main input data sources and data generation tools. More detailed source attribution can be found in the input spreadsheets themselves.

Input | Source
-- | ----
Technology and carrier definitions | [Kubiczek P. (2025). Technology and carrier definitions for PyPSA-PL model. Instrat.](https://docs.google.com/spreadsheets/d/1oM4T3LirR-XGO1fQ_KhiuQXW8t3I4AKj8q0n8P0s-aE)
Technological and cost assumptions | [Kubiczek P., Żelisko W. (2025). Technological and cost assumptions for PyPSA-PL model. Instrat.](https://docs.google.com/spreadsheets/d/1P-CGOaUUJt3J-6DfelAx5ilRSy0r2gCyJp_ZeHu1wbI)
Installed capacity assumptions | [Kubiczek P. (2025). Installed capacity assumptions for PyPSA-PL model. Instrat.](https://docs.google.com/spreadsheets/d/1fwosQK76x_FoXRSI6tphexjMchXSIX0NqAfHNCDI_BA)
Annual energy flow assumptions | [Kubiczek P. (2025). Annual energy flow assumptions for PyPSA-PL model. Instrat.](https://docs.google.com/spreadsheets/d/1OWm53wIPTVJf0PGUrUxhjpzfVJgyMhwdBLg5cuRzvZY)
Capacity utilisation assumptions | [Kubiczek P. (2025). Capacity utilisation assumptions for PyPSA-PL model. Instrat.](https://docs.google.com/spreadsheets/d/1OTZmzscUlB6uxuaWvN5Et1qpixFMubnh2m4-qbZD7rk)
Installed capacity potential and maximum addition assumptions | [Kubiczek P. (2025). Installed capacity potential and maximum addition assumptions for PyPSA-PL model. Instrat.](https://docs.google.com/spreadsheets/d/1z2pfJ6VwmjsgGgChJexISJZ-OYlrVllMUe1Q14Y5eR0)
Electricity final use time series | ENTSO-E. (2023). Total Load—Day Ahead / Actual. Transparency Platform. https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show
Wind and solar PV availability time series | Hofmann, F., Hampp, J., Neumann, F., Brown, T., Hörsch, J. (2021). atlite: A Lightweight Python Package for Calculating Renewable Power Potentials and Time Series. Journal of Open Source Software, 6, 3294. https://doi.org/10.21105/joss.03294 
Temperature data used to infer space heating demand and heat pump COP time series | IMGW. (2023). Dane publiczne. Instytut Meteorologii i Gospodarki Wodnej. https://danepubliczne.imgw.pl/
Daily space heating demand time series | Ruhnau, O., Muessel, J. (2023). When2Heat Heating Profiles. Open Power System Data. https://doi.org/10.25832/when2heat/2023-07-27 
Traffic data used to infer light vehicle mobility and BEV charging time series | GDDKiA. (2023). Stacje Ciągłych Pomiarów Ruchu (SCPR). Generalna Dyrekcja Dróg Krajowych i Autostrad. https://www.gov.pl/web/gddkia/stacje-ciaglych-pomiarow-ruchu
Power transmission grid | JAO. (2024). Static Grid Model (6th release). Joint Allocation Office. https://www.jao.eu/static-grid-model 


## Publications and full datasets

Here you can find the list of publications based on the PyPSA-PL results and links to the full datasets stored in Zenodo.


* Kubiczek P., Nowak K., Smoleń M. (2025). Sieci na miarę. Koszty sieciowe a opłacalność OZE w Polsce do 2040 r. Instrat Policy Paper 03/2025. (awaiting translation into English)
 https://instrat.pl/sieci-a-oze [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17779551.svg)](https://doi.org/10.5281/zenodo.17779551)
* Kubiczek P., Smoleń M. (2024). Three challenging decades. Scenario for the Polish energy transition out to 2050. Instrat Policy Paper 03/2024. https://instrat.pl/three-challenging-decades/ [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13946776.svg)](https://doi.org/10.5281/zenodo.13946776)
* Kubiczek P., Smoleń M., Żelisko W. (2023). Poland approaching carbon neutrality. Four scenarios for the Polish energy transition until 2040. Instrat Policy Paper 06/2023. https://instrat.pl/poland-2040/ [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10246018.svg)](https://doi.org/10.5281/zenodo.10246018)
* Kubiczek P. (2023). Baseload power. Modelling the costs of low flexibility of the Polish power system. Instrat Policy Paper 04/2023. https://instrat.pl/baseload-power/ [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8263172.svg)](https://zenodo.org/record/8263172)
* Kubiczek P., Smoleń M. (2023). Poland cannot afford medium ambitions. Savings driven by fast deployment of renewables by 2030. Instrat Policy Paper 03/2023. https://instrat.pl/pypsa-march-2023/ [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7784931.svg)](https://doi.org/10.5281/zenodo.7784931)


## Acknowledgements

The current version of PyPSA-PL is a successor of the [PyPSA-PL v1](https://github.com/instrat-pl/pypsa-pl/tree/v1) developed by Instrat in 2021. The following publications were based on the PyPSA-PL v1 results:

* Czyżak, P., Wrona, A. (2021). Achieving the goal. Coal phase-out in Polish power sector. Instrat Policy Paper 01/2021. https://instrat.pl/coal-phase-out
* Czyżak, P., Sikorski, M., Wrona, A. (2021). What’s next after coal? RES potential in Poland. Instrat Policy Paper 06/2021. https://instrat.pl/res-potential
* Czyżak, P., Wrona, A., Borkowski, M. (2021). The missing element. Energy security considerations. Instrat Policy Paper 09/2021. https://instrat.pl/energy-security


## License

The code is released under the [MIT license](LICENSE). The input and output data are released under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

&copy; Fundacja Instrat 2025

<a href="https://instrat.pl/en/"><img src="docs/instrat.png" width="200"></a>
