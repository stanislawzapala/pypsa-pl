from pypsa.statistics.expressions import get_weightings
from pypsa.utils import pass_empty_series_if_keyerror


def make_opex_calculator(cost_attr="marginal_cost", name="Operational Expenditure"):
    # Based on https://github.com/PyPSA/PyPSA/blob/565e7b4/pypsa/statistics/expressions.py#L581-L634
    def opex(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        aggregate_across_components=False,
        groupby="carrier",
        at_port=False,
        bus_carrier=None,
        nice_names=None,
    ):
        """
        Calculate the operational expenditure in the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated
            using snapshot weightings. With False the time series is given in currency/hour. Defaults to 'sum'.
        """

        @pass_empty_series_if_keyerror
        def func(n, c, port):
            if c in n.branch_components:
                p = n.dynamic(c).p0
            elif c == "StorageUnit":
                p = n.dynamic(c).p_dispatch
            else:
                p = n.dynamic(c).p

            opex = p * n.get_switchable_as_dense(c, cost_attr)
            weights = get_weightings(n, c)
            return self._aggregate_timeseries(opex, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = name
        df.attrs["unit"] = "currency"
        return df

    return opex
