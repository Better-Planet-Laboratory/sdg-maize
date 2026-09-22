from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import itertools

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

theme_bw = 'scripts/theme_bw.mplstyle'
plt.style.use(theme_bw)

def printdf(df: pd.DataFrame, num_rows: int=5, ignore_geometry: bool = True):
    if hasattr(df, 'geometry') and ignore_geometry:
        df = df[[col for col in df.columns if col!='geometry']]
    print(df.head(num_rows).to_string())

# region Data

# region Administrative boundaries - Rwanda Villages
villages = gpd.read_file('data/WB_NISR_2018/Village.shp')
villages.to_crs(4326, inplace=True)

def correct_village_geom(geom):

    geom_type = geom.geom_type

    village_xy = ([geom.exterior.coords.xy] if geom_type == 'Polygon'
                  else [p.exterior.coords.xy for p in
                        list(geom.geoms)] if geom_type == 'MultiPolygon'
    else print('Geometry is neither a polygon or multipolygon'))

    village_xy_concat = [zip(lst[0], lst[1]) for lst in village_xy]
    village_xy_list = [[[x, y] for x, y in pair] for pair in village_xy_concat]

    poly = (Polygon(village_xy_list[0]) if geom_type == 'Polygon'
            else MultiPolygon([Polygon(lst) for lst in village_xy_list]) if geom_type == 'MultiPolygon'
    else print('Geometry is neither a polygon or multipolygon'))

    return poly

corrected_geoms = [correct_village_geom(g) for g in villages.geometry]  # Remove empty z-axis from coordinates
villages.geometry = corrected_geoms
villages['Village_ID'] = pd.to_numeric(villages['Village_ID'])
# endregion

# region Administrative boundaries - Rwanda Districts
districts = villages[['Distr_ID', 'District', 'geometry', 'Prov_ID', 'Province']].dissolve(by='Distr_ID', aggfunc="first")
# endregion

# region FAO maize
# https://www.fao.org/faostat/en/#data/QCL
fao = pd.read_csv('data/FAOSTAT_QCL_Rwanda_Maize.csv')
fao_maize = fao[['Element', 'Year', 'Value']].pivot(
    index='Year', columns='Element', values='Value').reset_index().rename(
    columns={
        'Year': 'year',
        'Area harvested': 'maize_area_ha',
        'Production': 'maize_prodx_tonne',
        'Yield': 'maize_yield_100gha'})
fao_maize['maize_yield_kgha'] = fao_maize['maize_yield_100gha'] * 100 / 1000
# endregion

# region Maize predictions
maize_files = list(Path('data/').glob('maize_stats_by_village_maize*'))

village_area = pd.read_csv('data/maize_stats_by_village_area_ha.csv')

for i,f in enumerate(maize_files):
    if i == 0:
        maize_village_stats = pd.read_csv(f)
    else:
        maize_village_stats = maize_village_stats.merge(pd.read_csv(f))

def add_vars_from_season(df):
    df['season_ab'] = df['season'].apply(lambda x: x.split("_")[-1])
    df['year'] = df['season'].apply(lambda x: x.split("_")[0])
    return df

maize_village_stats = add_vars_from_season(maize_village_stats)
maize_village_stats = maize_village_stats.merge(village_area[['Village_ID', 'area_ha_sum']], how='left', on=['Village_ID'])
maize_village_stats['maize_area_pct'] = maize_village_stats['maize_area_ha_sum'] / maize_village_stats['area_ha_sum']
maize_village_stats['maize_prodx_tonne'] = maize_village_stats['maize_prodx_kg_sum'] / 1000  # kg to tonne

seasons = maize_village_stats['season'].unique().tolist()
# endregion

# region Annual maize predictions
maize_village_stats_annual = maize_village_stats.copy().groupby([
    'Province', 'Prov_ID', 'District', 'Distr_ID', 'Sector', 'Sector_ID', 'Cell', 'Cell_ID',
    'Name', 'Village_ID', 'year'], as_index=False, dropna=False).agg({
    'maize_yield_kgha_mean': 'max',  # peak yield between seasons
    'maize_prodx_tonne': 'sum'}  # yield summed across seasons
)
maize_village_stats_annual['maize_yield_kgha_mean'] = maize_village_stats_annual['maize_yield_kgha_mean'].fillna(0)

# Mean annual yield weighted by area
maize_village_stats_annual = pd.merge(
    maize_village_stats_annual,
   maize_village_stats.groupby(['Village_ID', 'year'], as_index=False, dropna=False).apply(
    lambda df: np.nan if all(df['maize_area_ha_sum'] == 0) else np.average(df['maize_yield_kgha_mean'].fillna(0), weights=df['maize_area_ha_sum'])).rename(
        columns={None: 'maize_yield_kgha_mean_weighted'})
)

# Rename columns so that weighted average is used as primary outcome
maize_village_stats_annual.rename(columns={'maize_yield_kgha_mean': 'maize_yield_kgha_mean_peak',
                                           'maize_yield_kgha_mean_weighted': 'maize_yield_kgha_mean'}, inplace=True)

# Clean year variable
maize_village_stats_annual['year'] = maize_village_stats_annual['year'].astype(int)

# Sort by village
maize_village_stats_annual.sort_values(['Village_ID', 'year'], inplace=True)
# endregion

# endregion

# region National baseline

# Establish baselines and required rate of change
baseline_year = 2015
sdg_end_year = 2030
year_chg_pt = 2024
year_chg_pt_name = 'y' + str(year_chg_pt)

fao_yield_baseline = fao_maize[fao_maize.year == baseline_year]['maize_yield_kgha'].item()
fao_prodx_baseline = fao_maize[fao_maize.year == baseline_year]['maize_prodx_tonne'].item()

# Producer quantiles
num_of_quantiles = 10
qntl_lowest_and_highest = [0, num_of_quantiles - 1]


# Growth rate
req_ratio = 2  # i.e. doubling

def calculate_linear_growth_slope(start, target, periods):
    return (target - start) / periods

req_slope_lin_chg_yield = calculate_linear_growth_slope(start=fao_yield_baseline, target=fao_yield_baseline*2, periods=(2030 - baseline_year))
req_slope_lin_chg_prodx = calculate_linear_growth_slope(start=fao_prodx_baseline, target=fao_prodx_baseline*2, periods=(2030 - baseline_year))

print(f'Required rate of growth (linear), kg/ha: {req_slope_lin_chg_yield:.2f}' )
print('Required rate of growth (linear), tonnes:', req_slope_lin_chg_prodx)

# FAO annual rate of change
fao_maize_since_baseline = fao_maize[fao_maize.year >= baseline_year]
print('Actual FAO rate of change:', fao_maize_since_baseline[['maize_yield_kgha', 'maize_prodx_tonne']].pct_change().mean())

# Predictions annual rate of change
print('Actual predicted rate of change:\n',
     maize_village_stats_annual.groupby(['year'])[['maize_yield_kgha_mean', 'maize_prodx_tonne']].mean().pct_change().mean())
# endregion

# region National yields
maize_annual_mean_by_source = pd.concat([
    fao_maize_since_baseline[['year', 'maize_yield_kgha', 'maize_prodx_tonne']].rename(columns={'maize_yield_kgha': 'maize_yield_kgha_mean'}),
    pd.concat([
        maize_village_stats_annual[['year', 'maize_yield_kgha_mean']].replace(0, np.nan).groupby('year')['maize_yield_kgha_mean'].mean(),  # weighted annual mean among villages with any maize
        maize_village_stats_annual.groupby('year')['maize_prodx_tonne'].sum()
    ], axis=1).reset_index()
], keys=['FAO', 'Fankhauser et al.']).reset_index(level=0, names=['source'])
maize_annual_mean_by_source['year'] = maize_annual_mean_by_source['year'].apply(lambda x: int(x))

# Add required rate of change based on linear growth
maize_annual_lin_growth_req = pd.DataFrame({'source': 'SDG 2.3', 'year': np.arange(baseline_year, 2030+1, 1), 'maize_yield_kgha_mean': fao_yield_baseline, 'maize_prodx_tonne': fao_prodx_baseline})
for i in range(1, len(maize_annual_lin_growth_req)):
    maize_annual_lin_growth_req.loc[i, 'maize_yield_kgha_mean'] = maize_annual_lin_growth_req.loc[i-1, 'maize_yield_kgha_mean'] + req_slope_lin_chg_yield
    maize_annual_lin_growth_req.loc[i, 'maize_prodx_tonne'] = maize_annual_lin_growth_req.loc[i-1, 'maize_prodx_tonne'] + req_slope_lin_chg_prodx

maize_annual_mean_by_source = pd.concat([maize_annual_mean_by_source, maize_annual_lin_growth_req])
# endregion

# region Figure 1 - National maize yield trends in Rwanda compared to SDG 2.3 targets

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
g = sns.lineplot(maize_annual_mean_by_source.drop('maize_prodx_tonne', axis=1), x='year', y='maize_yield_kgha_mean',
    hue='source', palette=['#1f77b4', '#ff7f0e', '#2ca02c'], ax=ax)

# Clean up axes
ax.set_title("Maize Yield (kg/ha)")
ax.set_xticks(np.arange(2015, 2031, 1).tolist())
[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 3 != 0]

ax.set_xlabel('')
ax.set_ylabel('')
sns.move_legend(g, loc='lower center', title=None, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
plt.show()

print('Yield gap in 2023:',
      maize_annual_mean_by_source[(maize_annual_mean_by_source['year'] == 2023) & (maize_annual_mean_by_source['source'] == 'SDG 2.3')]['maize_yield_kgha_mean'].item() - maize_annual_mean_by_source[(maize_annual_mean_by_source['year'] == 2023) & (maize_annual_mean_by_source['source'] == 'Fankhauser et al.')]['maize_yield_kgha_mean'].item()
      )
# endregion

# region Village yields

# Projections - Linear regression by village
def lm_yield_by_district(district_id, observation_years):

    data_subset = maize_village_stats_annual[maize_village_stats_annual['Distr_ID'] == district_id]
    data_subset = data_subset[data_subset['year'].isin(observation_years)]  # don't include 2024 b/c preliminary
    data_subset = data_subset[data_subset['Village_ID'].isin(
        data_subset
        .groupby('Village_ID')
        .apply(lambda x: all(np.isnan(x['maize_yield_kgha_mean'])), include_groups=False)
        .loc[lambda x: x == False].reset_index()['Village_ID'])]  # remove villages that never produce maize

    model = smf.ols(formula='maize_yield_kgha_mean ~ 0 + year*C(Village_ID)',   # remove intercept, without which some districts do not converge (does not affect prediction)
                    data=data_subset)
    fit = model.fit()

    newdata = pd.DataFrame(
        data=itertools.product(
            data_subset['Village_ID'].unique(),
            [baseline_year, year_chg_pt, sdg_end_year]),
        columns=['Village_ID', 'year'])

    predictions = fit.get_prediction(newdata)
    prediction_intervals = predictions.summary_frame(alpha=0.05)

    df_prediction = pd.concat([newdata, prediction_intervals], axis=1)

    return df_prediction

district_ids = maize_village_stats_annual['Distr_ID'].unique()

predictions_by_district = [lm_yield_by_district(id, observation_years=np.arange(2019, 2023+1)) for id in district_ids]

predicted_lm_chg = pd.concat(predictions_by_district)

# Censor yields to 90 kg/ha (one hermetic bag, i.e. total crop failure)
rate_names = ['mean', 'obs_ci_lower', 'obs_ci_upper']

maize_village_growth_rates = predicted_lm_chg.copy()[
    ['Village_ID', 'year'] + rate_names].mask(predicted_lm_chg < 90, 90)

# Attach slope, ratio, and meets SDG to each dataset
def clean_growth_rate_dfs(df, prediction_var):

    # Long to wide
    df_wide = df.pivot(index='Village_ID', columns='year', values=prediction_var).rename(columns=dict(zip(maize_village_growth_rates['year'].unique(), ['y' + y for y in maize_village_growth_rates['year'].unique().astype(str)]))).reset_index()

    # Slope
    df_wide['slope'] = (df_wide['y2030'] - df_wide['y2015']) / (sdg_end_year - baseline_year)

    # Ratio
    df_wide['ratio'] = df_wide['y2030'] / df_wide['y2015']

    # Determine if ratio meets SDG, i.e. >= 2
    df_wide['meets_sdg'] = df_wide['ratio'] >= 2

    return df_wide

mean_growth, lpi_growth, upi_growth = [clean_growth_rate_dfs(df=maize_village_growth_rates, prediction_var=var) for var in rate_names]

# Highest projected baseline yield
print(f"Highest projected baseline yield: {mean_growth['y2015'].max():.2f}",
      f"95% PI: {lpi_growth['y2015'].max():.2f} - {upi_growth['y2015'].max():.2f}")

# Projected mean growth compared to goal
print(f"Projected mean in 2030: {mean_growth['y2030'].mean():.2f}", )
print(f"Projected gap in 2030: {fao_yield_baseline*2 - mean_growth['y2030'].mean():.2f}")
# endregion

# region Figure 2 + Fig S3 & S4 - Rwandan villages on and off track to meet SDG 2.3

def map_predicted_growth(df):

    # Figure
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6), dpi=300, constrained_layout=True)

    # Data for map
    data_map_growth = df.copy().merge(villages[['Village_ID', 'geometry']], how='right')
    data_map_growth = gpd.GeoDataFrame(data_map_growth, geometry='geometry', crs='EPSG:4326')

    # Map
    data_map_growth.dropna().plot(column='ratio', cmap='Spectral', vmin=0, vmax=2,
                         edgecolor="face", linewidth=0.4, ax=ax1,
                         legend=True, legend_kwds={"label": "Growth rate between 2015 and 2030", "orientation": "horizontal"},
                                  )

    # Histogram
    data_for_hist_growth = df.copy()
    bin_unit = 0.25

    ratio_bins = np.arange(0, 3 + bin_unit, bin_unit).tolist()
    data_for_hist_growth['ratio_clipped'] = np.clip(data_for_hist_growth['ratio'], ratio_bins[0], ratio_bins[-1])

    hue_bins = np.arange(0, req_ratio + bin_unit, bin_unit).tolist() + [np.inf]
    data_for_hist_growth['ratio_hue'] = pd.cut(x=data_for_hist_growth['ratio'], bins=hue_bins,
                                               include_lowest=True, right=False,
                                               labels=False)

    sns.histplot(data=data_for_hist_growth, x='ratio_clipped', bins=ratio_bins + [ratio_bins[-1] + bin_unit], stat='count', hue='ratio_hue', palette='Spectral', alpha=1, legend=False, ax=ax2)
    ax2.set_xticks(ratio_bins)
    ax2.set_xticklabels(ratio_bins[:-1] + [f'{ratio_bins[-1]}+'])
    ax2.axvline(req_ratio,  ls='--', c='black')
    ax2.set_xlabel('Ratio')
    ax2.set_ylabel("Number of villages")
    ax2.text(x=req_ratio, y=0.99, s='SDG 2.3',
            ha='right', va='top', rotation=90, color='black', transform=ax2.get_xaxis_transform())

    return plt

# Fig 2 - Map and histogram of predicted growth rate - Mean
map_mean_growth = map_predicted_growth(mean_growth)
map_mean_growth.show()

# S3 Fig - Map and histogram of predicted growth rate - LPI
map_lpi_growth = map_predicted_growth(lpi_growth)
map_lpi_growth.show()

# S4 Fig - Map and histogram of predicted growth rate - UPI
map_upi_growth = map_predicted_growth(upi_growth)
map_upi_growth.show()

# endregion

# region Policy scenarios
def calculate_linear_change(t1, y_t1, slope, tI):
    y_tI = y_t1 + slope * (tI - t1)
    return y_tI

def solve_for_years(t0, y_t0, t1, y_t1, y_tI):
    m = (y_t1 - y_t0) / (t1 - t0)
    return (y_tI - y_t0) / m

def summarize_sdg_outcomes(df):

    # Wide to long
    df_long = pd.wide_to_long(df[['Village_ID', 'y2015', year_chg_pt_name, 'y2030']], stubnames="y", i='Village_ID', j='year').reset_index()

    # Annual dataset
    df_annual = df_long.groupby('year')['y'].mean().reset_index()

    # Calculate baseline yield in predictions
    baseline_yield = df_annual[df_annual['year'] == baseline_year]['y'].item()

    # Percent of national progress
    pct_of_natl_progress = (
            (df_annual[df_annual['year'] == sdg_end_year][
                 'y'].item() - baseline_yield)
            / baseline_yield)

    # Number of years to meet SDG
    yrs_to_goal = solve_for_years(
            t0=year_chg_pt,
            y_t0=df_annual[df_annual['year'] == year_chg_pt]['y'].item(),
            t1=sdg_end_year,
            y_t1=df_annual[df_annual['year'] == sdg_end_year]['y'].item(),
            y_tI=baseline_yield * req_ratio)
    add_yrs_to_goal = yrs_to_goal - (sdg_end_year - year_chg_pt)
    # If growth rate b/w 2023 and 2030 is negative then set additional years to Inf
    m = (df_annual[df_annual['year'] == sdg_end_year]['y'].item() - df_annual[df_annual['year'] == year_chg_pt]['y'].item()) / (sdg_end_year - year_chg_pt)
    add_yrs_to_goal = np.inf if m < 0 else add_yrs_to_goal

    # Progress by village
    df_ratio = df_long[df_long['year'].isin([
        df_long['year'].min(),
        df_long['year'].max()])
    ].groupby('Village_ID')['y'].apply(
        lambda x: (x / x.shift()).tail(1)).droplevel(1).reset_index().rename(
        columns={'y': 'ratio'})
    pct_of_village_progress = (df_ratio['ratio'] >= req_ratio).mean()

    # Equity ratio
    df_qntls = df_long.copy()[df_long['year'] == sdg_end_year]
    if (df_qntls['y'].round(5).nunique() == 1):  # all the same mean implies equity
        equity_ratio = 1.0
    else:
        df_qntls['qntl'] = df_qntls['y'].transform(
            lambda x: pd.qcut(x, num_of_quantiles, labels=False, duplicates='drop')).astype(int)

        equity_ratio = df_qntls[df_qntls['qntl'].isin([df_qntls['qntl'].min(), df_qntls['qntl'].max()])].groupby('qntl')[
            'y'].mean().reset_index().apply(lambda x: x / x.shift()).dropna()[
            'y'].item()

    # Max growth rate after change point
    df_growth = df_long[df_long['year'].isin([year_chg_pt, sdg_end_year])]
    max_growth_rate = df_growth.groupby('Village_ID').apply(
        lambda df: (df['y'] - df.shift()['y'])
                   / (df['year'] - df.shift()['year']), include_groups=False).max()

    # Output dictionary
    return {'pct_of_natl_progress': pct_of_natl_progress,
            'add_yrs_to_goal': add_yrs_to_goal,
            'pct_of_village_progress': pct_of_village_progress,
            'equity_ratio': equity_ratio,
            'max_growth_rate': max_growth_rate
    }

# region Scenario 1 - Current trajectory
sc1_current = pd.concat([pd.DataFrame([summarize_sdg_outcomes(df)]) for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)
# endregion

# region Scenario 2 - National SDG (Uniform slope)
def apply_uniform_slope(df):

    # Calculate baseline yield in predictions
    baseline_yield = df['y2015'].mean().item()

    # Calculate uniform slope
    uniform_slope = (
        ((baseline_yield * req_ratio)  # target natl yield
        - df[year_chg_pt_name].mean())  # mean in chg pt year
        / (sdg_end_year-year_chg_pt)  # years to SDG
    )

    # Apply uniform slope
    df_uniform = df.copy()[['Village_ID', 'y2015', year_chg_pt_name]]
    df_uniform['slope'] = uniform_slope
    df_uniform['y2030'] = df_uniform.apply(
    lambda df: calculate_linear_change(
        t1=year_chg_pt, y_t1=df[year_chg_pt_name],
        slope=uniform_slope,
        tI=sdg_end_year), axis=1)

    # Return dataset with uniform slope applied
    return df_uniform

sc2_uniform = pd.concat([pd.DataFrame([summarize_sdg_outcomes(apply_uniform_slope(df))]) for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)
# endregion

# region Scenario 3 - Village SDG (double yield in all villages)
def apply_double_yield(df):

    # Subset dataset
    df_double = df.copy()[
    ['Village_ID', 'y2015', year_chg_pt_name]]

    # Apply double rate
    df_double['y2030'] = df_double['y2015'] * req_ratio
    df_double['slope'] = df_double.apply(
        lambda x: calculate_linear_growth_slope(start=x[year_chg_pt_name], target=x['y2030'], periods=(sdg_end_year-year_chg_pt)),
        axis=1)

    return df_double

sc3_double = pd.concat([pd.DataFrame([summarize_sdg_outcomes(apply_double_yield(df))]) for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)
# endregion

# region Scenario 4 - Equity
def apply_equity(df, target):

    # Subset and copy dataframe
    df_equity = df.copy()[['Village_ID', 'y2015', year_chg_pt_name, 'y2030']]

    # Quantiles in 2030
    df_equity['qntl_2030'] = df_equity[['y2030']].transform(lambda x: pd.qcut(x, num_of_quantiles, labels=False)).astype(int)

    # Define target
    if target == 'equity':
        df_q90_asis = df_equity[df_equity['qntl_2030'] == qntl_lowest_and_highest[1]]
        target_yield = df_q90_asis['y2030'].mean()  # 2030 mean in highest quantile
    elif target == 'equity_and_natl_sdg':
        baseline_yield = df['y2015'].mean().item()
        target_yield = baseline_yield*req_ratio,  # mean meets SDG
    elif target == 'equity_and_village_sdg':
        target_yield = df_equity['y2015'].max()*req_ratio,  # all villages double
    else:
        print("Desired target must equal one of: 'equity', 'equity_and_natl_sdg', or 'equity_and_village_sdg'")
    target_yield = target_yield[0] if isinstance(target_yield, tuple) else target_yield

    # Calculate slope needed to achieve equity
    df_equity['slope'] = df_equity.apply(
    lambda df: calculate_linear_growth_slope(
        target=target_yield,
        start=df[year_chg_pt_name],
        periods=sdg_end_year-year_chg_pt
    ), axis=1)

    # Apply equity slope
    df_equity['y2030'] = df_equity.apply(
    lambda df: calculate_linear_change(
        t1=year_chg_pt, y_t1=df[year_chg_pt_name],
        slope=df['slope'],
        tI=sdg_end_year), axis=1)

    # Return equity dataframe
    return df_equity

sc4_equity = pd.concat([pd.DataFrame([
    summarize_sdg_outcomes(apply_equity(df, target='equity'))])
    for df in [mean_growth, lpi_growth, upi_growth]],
    axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)
# endregion

# region Scenario 5: Equity + National SDG
sc5_equity_and_natl_sdg = pd.concat([pd.DataFrame([
    summarize_sdg_outcomes(apply_equity(df, target='equity_and_natl_sdg'))])
    for df in [mean_growth, lpi_growth, upi_growth]],
    axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)
# endregion

# region Supplement Scenario 6: Max achieved growth rate
max_growth = maize_village_stats_annual.groupby(
    'Village_ID')['maize_yield_kgha_mean'].apply(
    lambda x: x -x.shift()
).droplevel(1).reset_index().groupby(
    'Village_ID', as_index=False)['maize_yield_kgha_mean'].max().rename(
    columns={'maize_yield_kgha_mean': 'slope'})

# Add in first observation
max_growth = pd.merge(
    max_growth,
    maize_village_stats_annual[maize_village_stats_annual['year'] == 2019][['Village_ID', 'maize_yield_kgha_mean']].rename(columns={'maize_yield_kgha_mean': 'y2019'})
)

# Baseline yield in 2015
max_growth['y2015'] = max_growth.apply(
    lambda df: max(90,  # censor to 90
                   calculate_linear_change(t1=2019, y_t1=df['y2019'], slope=df['slope'], tI=2015)
                   ), axis=1
)

# Projected yield in change point year
max_growth[year_chg_pt_name] = max_growth.apply(
    lambda df: max(90,
                   calculate_linear_change(t1=2019, y_t1=df['y2019'], slope=df['slope'], tI=year_chg_pt)
                   ), axis=1
)
# Remove villages that never produce maize yield
max_growth = max_growth[max_growth['Village_ID'].isin(
    maize_village_stats_annual
    .groupby('Village_ID')
    .apply(lambda x: all(np.isnan(x['maize_yield_kgha_mean'])), include_groups=False)
    .loc[lambda x: x == False].reset_index()['Village_ID'])]

def apply_max_obs_growth(df):

    df_max = df.copy()[['Village_ID', 'y2015', year_chg_pt_name]]

    # Apply max slope
    df_max = df_max.merge(max_growth[['Village_ID', 'slope']])
    df_max['y2030'] = df_max.apply(
    lambda df: np.nan if np.isnan(df['slope']) else max(
        90,
        calculate_linear_change(
        t1=year_chg_pt, y_t1=df[year_chg_pt_name],
        slope=df['slope'],
        tI=sdg_end_year)), axis=1
    )

    return df_max

sc6_max_growth = pd.concat([pd.DataFrame([summarize_sdg_outcomes(apply_max_obs_growth(df).dropna())]) for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)
# endregion

# region Supplement Scenario 7: Equity + National SDG + Village SDG
sc7_equity_and_village_sdg = pd.concat([pd.DataFrame([
    summarize_sdg_outcomes(apply_equity(df, target='equity_and_village_sdg'))])
    for df in [mean_growth, lpi_growth, upi_growth]],
    axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)
# endregion

# endregion

# region Table 1 + S2 Table - National and village-level outcomes towards SDG 2.3 to double productivity for maize under potential scenarios
scenario_names = ['Sc1: Current', 'Sc2: National SDG', 'Sc3: Village SDG',
                 'Sc4: Equitable', 'Sc5: Equitable + Natl SDG',
                 'Sc6: Max achieved growth rate', 'Sc7: Equitable + Village SDG']

tbl_scenarios = pd.concat([sc1_current, sc2_uniform, sc3_double,
                           sc4_equity, sc5_equity_and_natl_sdg,
                           sc6_max_growth, sc7_equity_and_village_sdg],
                          axis=0,
                          keys=scenario_names, names=['Scenario'])
# endregion

# region Supplementary Material

# region S1 Fig - Sub-national inequality in maize yields.

# Data

# Define quantiles
maize_village_stats_annual['prodx_qntl'] = (maize_village_stats_annual
                                            [maize_village_stats_annual['maize_prodx_tonne'] > 0]  # create quantiles only among villages with any maize prodx
                                            .groupby(['year'])
                                            ['maize_prodx_tonne'].transform(lambda x: pd.qcut(x, num_of_quantiles, labels=False)).astype(int))
maize_village_stats_annual['yield_qntl'] = (maize_village_stats_annual
                                            .dropna(subset='maize_yield_kgha_mean')  # create quantiles only among villages with any maize prodx/yield
                                            .groupby(['year'])
                                            ['maize_yield_kgha_mean'].transform(lambda x: pd.qcut(x, num_of_quantiles, labels=False)).astype(int))

# Define baseline quantiles
qntls_2019 = maize_village_stats_annual.copy()[maize_village_stats_annual.year == 2019].rename(
    columns={'prodx_qntl': 'prodx_qntl_2019', 'yield_qntl': 'yield_qntl_2019'})
maize_village_stats_annual = maize_village_stats_annual.merge(
    qntls_2019[['Village_ID', 'prodx_qntl_2019', 'yield_qntl_2019']], how='left')

# Mean yield by quantile
yield_ratio_lowest_and_highest = maize_village_stats_annual[maize_village_stats_annual.yield_qntl_2019.isin(qntl_lowest_and_highest)].groupby(
    ['year', 'yield_qntl_2019'])['maize_yield_kgha_mean'].agg(
    mean='mean', min='min', max='max',
    q025=(lambda x: x.quantile(0.025)),
    q975=(lambda x: x.quantile(0.975))).reset_index()

yield_ratio_lowest_and_highest['end_of_ag_year'] = yield_ratio_lowest_and_highest['year'].apply(lambda x: pd.to_datetime(str(x) + '-07-01'))

yield_ratio_lowest_and_highest['ratio'] = yield_ratio_lowest_and_highest.groupby('year')['mean'].transform(lambda x: x / x.shift())

# Prediction intervals
yield_ratio_lowest_and_highest['lpi'] = yield_ratio_lowest_and_highest.groupby('year').apply(lambda df: df['q025'] / df['q975'].shift(), include_groups=False).reset_index(0, drop=True)
yield_ratio_lowest_and_highest['upi'] = yield_ratio_lowest_and_highest.groupby('year').apply(lambda df: df['q975'] / df['q025'].shift(), include_groups=False).reset_index(0, drop=True)

yield_ratio_lowest_and_highest = yield_ratio_lowest_and_highest.reset_index().dropna()[['year', 'end_of_ag_year', 'ratio', 'lpi', 'upi']]

# Confidence intervals
def calculate_cis_for_ratio(df):

    data_q10 = df[df['yield_qntl_2019'] == 0]['maize_yield_kgha_mean'].dropna()
    data_q90 = df[df['yield_qntl_2019'] == 9]['maize_yield_kgha_mean'].dropna()
    pairwise_ratios = [y/x for x,y in itertools.product(data_q10, data_q90)]

    lci, uci = np.quantile(pairwise_ratios, [0.025, 0.975])

    return pd.DataFrame([{'lci': lci, 'uci': uci}])
ratio_cis = maize_village_stats_annual.groupby('year').apply(
    lambda df: calculate_cis_for_ratio(df), include_groups=False).droplevel(1).reset_index()

yield_ratio_lowest_and_highest = yield_ratio_lowest_and_highest.merge(ratio_cis)

print("Ratios:\n", yield_ratio_lowest_and_highest)

print("Mean yields by quantile/year:\n", maize_village_stats_annual[maize_village_stats_annual.yield_qntl_2019.isin(qntl_lowest_and_highest)]
      .groupby(['year', 'yield_qntl_2019'])['maize_yield_kgha_mean'].agg(mean='mean').reset_index())

# Figure
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6), dpi=300, constrained_layout=True)
palette_for_quantiles = sns.color_palette("mako", num_of_quantiles)

# Events
dates_of_events = pd.DataFrame(
    data={'event': ['COVID-19', 'Spike in Fertilizer $'],
          'date': pd.to_datetime([
              '2020-03-11', # Mar 11, 2020 WHO declares COVID-19 a pandemic; https://www.yalemedicine.org/news/covid-timeline
              '2022-04-01']) # Apr 2022 fertilizer price spikes; https://www.ers.usda.gov/amber-waves/2023/september/global-fertilizer-market-challenged-by-russia-s-invasion-of-ukraine
          })

# Plot - Ratio
sns.lineplot(yield_ratio_lowest_and_highest, x='end_of_ag_year', y='ratio', color='#fdbf6f', ax=ax1)
ax1.fill_between(x=yield_ratio_lowest_and_highest['end_of_ag_year'],
               y1=yield_ratio_lowest_and_highest['lci'],
               y2=yield_ratio_lowest_and_highest['uci'],
               color='#fdbf6f', alpha=0.2)

## x-axis labels
ax1.set_xlabel('')
ax1.set_xticks(yield_ratio_lowest_and_highest['end_of_ag_year'].unique())
ax1.set_xticklabels(np.arange(2019, 2025, 1).tolist())
# ax1.set_xticklabels([f'July\n{x}' for x in np.arange(2019, 2025, 1).tolist()])
ax1.set_title("Ratio in yields\nbetween lowest and highest percentiles")

## y-axis labels
ax1.set_ylabel('')
yield_ratio_plot_ticks = np.arange(1, 4.5, 0.5).tolist()
yield_ratio_plot_labels = ['Equal'] + yield_ratio_plot_ticks[1:-1] + ['Less equal']

ax1.set_yticks(yield_ratio_plot_ticks)
ax1.set_yticklabels(yield_ratio_plot_labels)

## Add annoations to ratio plot
[ax1.axvline(d, ymin=0, ymax=1, color='black', ls=':') for d in dates_of_events['date'].to_list()]
[ax1.text(d, 0.99, e, ha='right', va='top', rotation=90, color='black', transform=ax1.get_xaxis_transform()) for e,d in dates_of_events.values.tolist()]

# Plot - Mean yields by quantile
sns.lineplot(maize_village_stats_annual, x='year', y='maize_yield_kgha_mean',
             hue='yield_qntl_2019', palette=palette_for_quantiles, ax=ax2)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title("Maize yield (kg/ha)")

## Legend
sns.move_legend(ax2, "center right", bbox_to_anchor=(1.25, 0.5), ncol=1,
                title='Percentile in 2019',
                labels=['lowest'] +
                       [f'{x}-{np.arange(10, 100, 10).tolist()[i+1]}th' for i, x in enumerate(
                           np.arange(11, 90, 10).tolist())] +
                       ['highest'],
                frameon=False)

plt.show()

# endregion

# region S2, S5, S6 Fig - Projected yields over the SDG period for each development scenario with requisite growth rates and 2030 yields

# Data - Compare scenarios
# Datasets
df_current = pd.concat([mean_growth, lpi_growth, upi_growth], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)[['Village_ID', 'y2015', year_chg_pt_name, 'y2030', 'slope']]

df_uniform = pd.concat([apply_uniform_slope(df) for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)[['Village_ID', 'y2015', year_chg_pt_name, 'y2030', 'slope']]

df_double = pd.concat([apply_double_yield(df) for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)[['Village_ID', 'y2015', year_chg_pt_name, 'y2030', 'slope']]

df_equity = pd.concat([apply_equity(df, target='equity') for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)[['Village_ID', 'y2015', year_chg_pt_name, 'y2030', 'slope']]

df_equity_and_natl_sdg = pd.concat([apply_equity(df, target='equity_and_natl_sdg') for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)[['Village_ID', 'y2015', year_chg_pt_name, 'y2030', 'slope']]

df_equity_and_village_sdg = pd.concat([apply_equity(df, target='equity_and_village_sdg') for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)[['Village_ID', 'y2015', year_chg_pt_name, 'y2030', 'slope']]

df_max_growth = pd.concat([apply_max_obs_growth(df) for df in [mean_growth, lpi_growth, upi_growth]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)[['Village_ID', 'y2015', year_chg_pt_name, 'y2030', 'slope']]

# Format data for plots
data_for_scenario_plot = pd.concat([df_current, df_uniform, df_double,
                                    df_equity, df_equity_and_natl_sdg, df_equity_and_village_sdg, df_max_growth],
                                   axis=0, keys=scenario_names, names=['Scenario'])

data_for_scenario_plot = data_for_scenario_plot.reset_index()

# Plot only first 5 main scenarios
data_for_scenario_plot = data_for_scenario_plot[data_for_scenario_plot.Scenario.isin([
    'Sc1: Current', 'Sc2: National SDG', 'Sc3: Village SDG',
    'Sc4: Equitable', 'Sc5: Equitable + Natl SDG',
])]

data_for_scenario_plot = data_for_scenario_plot.melt(
    id_vars=['Village_ID', 'Scenario', 'level', 'slope'],
    value_vars=['y2015', year_chg_pt_name, 'y2030'],
    value_name='maize_yield_kgha_mean',
    var_name='year'
)

data_for_scenario_plot['year'] = data_for_scenario_plot['year'].transform(
    lambda x: int(''.join(filter(str.isdigit, x))))

# # Set scenario order
data_for_scenario_plot['Scenario'] = data_for_scenario_plot['Scenario'].astype(CategoricalDtype(categories=['Actual'] + scenario_names, ordered=True))

# Data - Time series

# Append actual observed yields
data_for_actual_scenario = maize_annual_mean_by_source[maize_annual_mean_by_source['source'].isin(['FAO', 'Fankhauser et al.'])].pivot(columns=['source'], values=['maize_yield_kgha_mean'], index='year').droplevel(0, axis=1)

data_for_actual_scenario = data_for_actual_scenario['FAO'].combine_first(data_for_actual_scenario['Fankhauser et al.']).reset_index().rename(columns={'FAO': 'maize_yield_kgha_mean'})

data_for_actual_scenario['Scenario'] = 'Actual'
pd.concat([data_for_actual_scenario]*3, ignore_index=True)
data_for_actual_scenario = pd.concat([data_for_actual_scenario]*3, keys=['mean', 'lpi', 'upi'], names=['level']).reset_index('level')

data_for_scenario_lineplot = pd.concat([data_for_actual_scenario, data_for_scenario_plot])

# Summarize to annual means
data_for_scenario_lineplot = data_for_scenario_lineplot.groupby(['Scenario', 'year', 'level'])['maize_yield_kgha_mean'].mean().reset_index()

data_for_scenario_lineplot = data_for_scenario_lineplot[data_for_scenario_lineplot['Scenario'].isin(
    ['Actual', 'Sc1: Current', 'Sc2: National SDG', 'Sc4: Equitable',
     # 'Sc7: Max achieved growth rate'
     ])].replace(
    ['Sc2: National SDG'], 'Meet SDG (Sc2, Sc3, Sc5)')  # All scenarios meeting the SDG have the same annual mean trend
data_for_scenario_lineplot['Scenario'] = data_for_scenario_lineplot['Scenario'].astype(CategoricalDtype(
    categories=['Actual', 'Sc1: Current', 'Sc4: Equitable',
                # 'Sc7: Max achieved growth rate',
                'Meet SDG (Sc2, Sc3, Sc5)'], ordered=True))

# Figures
def plot_scenarios(level):

    valid = {'mean', 'lpi', 'upi'}
    if level not in valid:
        raise ValueError("plot_scenarios: level must be one of %r." % valid)

    fig, axs = plt.subplots(1,3, figsize=(18,6), dpi=300) # constrained_layout=True
    ax1, ax2, ax3 = axs

    # Panel 1 - Time series

    ## Main trend
    scenario_groups = data_for_scenario_lineplot['Scenario'].cat.categories.tolist()[1:] + ['Actual']
    scenario_palette = sns.color_palette('husl', len(scenario_groups))
    sns.lineplot(x='year', y='maize_yield_kgha_mean',
                 hue='Scenario', palette=scenario_palette, hue_order=scenario_groups, style='Scenario',
                 data=data_for_scenario_lineplot[data_for_scenario_lineplot["level"] == level], ax=ax1)

    ## Add SDG 2.3 mark
    target_yield_pred = df_current[df_current.index == level]['y2015'].mean() * req_ratio
    ax1.axhline(target_yield_pred,  ls='--', c='darkgray')
    ax1.text(y=target_yield_pred + 40, x=baseline_year, s='SDG 2.3')

    ## Clean up axes
    ax1.set_xticks(np.arange(2015, 2031, 1).tolist())
    [l.set_visible(False) for (i,l) in enumerate(ax1.xaxis.get_ticklabels()) if i % 3 != 0]

    ax1.set_title("National mean")
    ax1.set_ylabel('Maize yield (kg/ha)')
    ax1.set_xlabel('')
    sns.move_legend(ax1, loc='lower center', title=None, ncol=2, bbox_to_anchor=(0.5, -0.3), frameon=False,
                    labels=['Sc1: Current', 'Sc4: Equitable',
                            # 'Sc7: Max achieved growth rate',
                            'Meet SDG\n(Sc2, Sc3, Sc5)', 'Actual'])

    # Panel 2 - Growth rates
    palette_for_boxplots = ['#f77189', '#c69432', '#37aabb', '#97a431', '#f45deb']

    data_for_scenario_boxplot = data_for_scenario_plot.copy()[~data_for_scenario_plot['Scenario'].isin([
        'Sc6: Max achieved growth rate'
        'Sc7: Equitable + Village SDG',
    ])]
    data_for_scenario_boxplot['Scenario'] = data_for_scenario_boxplot['Scenario'].cat.remove_unused_categories()
    data_for_scenario_boxplot = data_for_scenario_boxplot[data_for_scenario_boxplot['level'] == level]

    sns.boxplot(data_for_scenario_boxplot, ax=ax2, legend=False,
                    x='Scenario', y='slope', hue='Scenario', palette=palette_for_boxplots
                    # **{'boxprops': {'facecolor': 'none', 'edgecolor': 'black'}}
                    )
    ax2.tick_params(labelbottom=False)
    ax2.set_xlabel('')
    ax2.set_title('Annual growth rate')
    ax2.set_ylabel('Maize yield (kg/ha/year)')

    # Panel 3 - Yields by 2030
    sns.boxplot(data_for_scenario_boxplot[data_for_scenario_boxplot.year == 2030], ax=ax3, legend=True,
                    x='Scenario', y='maize_yield_kgha_mean', hue='Scenario', palette=palette_for_boxplots
                    # **{'boxprops': {'facecolor': 'none', 'edgecolor': 'black'}}
                    )
    ax3.tick_params(labelbottom=False)
    ax3.set_xlabel('')
    ax3.set_title("End year 2030")
    ax3.set_ylabel('Maize yield (kg/ha)')

    sns.move_legend(ax3, loc='lower right',
                    bbox_to_anchor=(0.9, -0.25),
                    title=None, ncol=5, frameon=False)

    return plt

# S2 Fig
scenarios_mean_growth = plot_scenarios('mean')
scenarios_mean_growth.show()

# S5 Fig
scenarios_lpi_growth = plot_scenarios('lpi')
scenarios_lpi_growth.show()

# S6 Fig
scenarios_upi_growth = plot_scenarios('upi')
scenarios_upi_growth.show()
# endregion

# region S7, S8, S9 Fig + S1 Table - Village-level maize yields in 2030 when observed data includes preliminary data from year 2024

# Data - Progress w/ prelim 2024 data
predictions_by_district_sup = [lm_yield_by_district(id, observation_years=np.arange(2019, 2024+1)) for id in district_ids]

predicted_lm_chg_sup = pd.concat(predictions_by_district_sup)

maize_village_growth_rates_sup = predicted_lm_chg_sup.copy()[
    ['Village_ID', 'year'] + rate_names].mask(predicted_lm_chg_sup < 90, 90)

mean_growth_sup, lpi_growth_sup, upi_growth_sup = [
    clean_growth_rate_dfs(df=maize_village_growth_rates_sup, prediction_var=var)
    for var in rate_names]

# Figures

# S7 Fig
map_mean_growth_sup = map_predicted_growth(mean_growth_sup)
map_mean_growth_sup.show()

# S8 Fig
map_lpi_growth_sup = map_predicted_growth(lpi_growth_sup)
map_lpi_growth_sup.show()

# S9 Fig
map_upi_growth_sup = map_predicted_growth(upi_growth_sup)
map_upi_growth_sup.show()

# S1 Table
sc1_current_sup = pd.concat([pd.DataFrame([summarize_sdg_outcomes(df)]) for df in [mean_growth_sup, lpi_growth_sup, upi_growth_sup]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)

sc2_uniform_sup = pd.concat([pd.DataFrame([summarize_sdg_outcomes(apply_uniform_slope(df))]) for df in [mean_growth_sup, lpi_growth_sup, upi_growth_sup]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)

sc3_double_sup = pd.concat([pd.DataFrame([summarize_sdg_outcomes(apply_double_yield(df))]) for df in [mean_growth_sup, lpi_growth_sup, upi_growth_sup]], axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)

sc4_equity_sup = pd.concat([pd.DataFrame([
    summarize_sdg_outcomes(apply_equity(df, target='equity'))])
    for df in [mean_growth_sup, lpi_growth_sup, upi_growth_sup]],
    axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)

sc5_equity_and_natl_sdg_sup = pd.concat([pd.DataFrame([
    summarize_sdg_outcomes(apply_equity(df, target='equity_and_natl_sdg'))])
    for df in [mean_growth_sup, lpi_growth_sup, upi_growth_sup]],
    axis=0, keys=['mean', 'lpi', 'upi'], names=['level']).droplevel(1)

tbl_scenarios_sup = pd.concat([sc1_current_sup, sc2_uniform_sup, sc3_double_sup,
                           sc4_equity_sup, sc5_equity_and_natl_sdg_sup],
                          axis=0,
                          keys=scenario_names[:5], names=['Scenario'])
# endregion

# region S10 Fig + S3 Table - Maximum maize yields observed and expected by scenario for each agro-ecological zone (AEZ)

# Data

# Projected CRS
proj_crs = "EPSG:32736"
villages_proj_crs = villages.to_crs(proj_crs)

# AEZ
aez = gpd.read_file('data/MINAGRI_AEZ_1980/zoneagro.shp')
aez.to_crs(proj_crs, inplace=True)

# Assign AEZ to village that covers most area (approx/relative)
intersect_village_and_aez = gpd.overlay(villages_proj_crs, aez[['ZONE_AGRO_', 'geometry']], how='intersection')
intersect_village_and_aez['area'] = intersect_village_and_aez.geometry.area
intersect_village_and_aez.sort_values(by='area', inplace=True)
intersect_village_and_aez.drop_duplicates(subset='Village_ID', keep='last', inplace=True)
intersect_village_and_aez.drop(columns=['area'], inplace=True)

# If village does not intersect AEZ (off coasts) then assign nearest
closest_aez_to_village = villages_proj_crs[~villages_proj_crs['Village_ID'].isin(intersect_village_and_aez['Village_ID'])].sjoin_nearest( aez[['ZONE_AGRO_', 'geometry']], how='left')

# Merge the two AEZ-Village datasets
villages_with_aez = pd.concat([intersect_village_and_aez, closest_aez_to_village], axis=0)
villages_with_aez.drop(columns=['index_right'], inplace=True)

# Calculate max observed yield by AEZ from 2019-2023
maize_village_stats_with_aez = maize_village_stats_annual[maize_village_stats_annual['year'] < 2024  # don't include 2024 b/c preliminary
].merge(villages_with_aez[['Village_ID', 'ZONE_AGRO_']])
max_yield_by_aez = maize_village_stats_with_aez.groupby('ZONE_AGRO_')['maize_yield_kgha_mean'].max().reset_index(
).rename(columns={'maize_yield_kgha_mean': 'max_obs_yield_kgha'})

# Calculate mean achievable yield if all villages max out
df_max_achievable = max_yield_by_aez.copy().merge(
    villages_with_aez.groupby('ZONE_AGRO_')['Village_ID'].count().reset_index())
max_achievable_yield_censored_by_aez = np.average(df_max_achievable['max_obs_yield_kgha'],
                                         weights=df_max_achievable['Village_ID'])

def censor_df_to_max_yield_in_aez(df):

    df_censored_by_aez = df.copy().merge(villages_with_aez[['Village_ID', 'ZONE_AGRO_']])
    df_censored_by_aez = df_censored_by_aez.merge(max_yield_by_aez)

    df_censored_by_aez[['y2015', year_chg_pt_name, 'y2030']] = df_censored_by_aez.apply(lambda df: df[['y2015', year_chg_pt_name, 'y2030']].apply(lambda x: min(x, df['max_obs_yield_kgha'])), axis=1)

    df_censored_by_aez.drop(
        ['slope', 'ZONE_AGRO_', 'max_obs_yield_kgha'],  # remove slope calculated for uncensored scenario and generated vars
        axis=1, inplace=True, errors='ignore')

    return df_censored_by_aez

def summarize_sdg_outcomes_sens(df):

    # Censor predictions to maximums by AEZ
    df_censored = censor_df_to_max_yield_in_aez(df)

    # Run outcome summary
    sc = summarize_sdg_outcomes(df_censored)

    # Target yield by 2030 based on censored yields
    target_yield = df_censored['y2015'].mean()*2

    # Replace number of years to meet SDG if goal surpasses AEZ limits
    # (i.e. Inf years b/c goal can never be met under these conditions)
    if target_yield > max_achievable_yield_censored_by_aez:
        sc.update(add_yrs_to_goal=np.inf)

    return sc

# Update equity dataset to represent all villages having yields of the lowest max achievable yield by AEZ
df_equity_sens = censor_df_to_max_yield_in_aez(df_equity.reset_index())
df_equity_sens['y2030'] = max_yield_by_aez['max_obs_yield_kgha'].min()
df_equity_sens.set_index('level', inplace=True)

# S10 Fig
def compute_max_yield_in_aez(df):
    df_with_aez = df.copy().reset_index().merge(villages_with_aez[['Village_ID', 'ZONE_AGRO_']])

    max_yield_2030_by_aez = df_with_aez.groupby(['level', 'ZONE_AGRO_'], as_index=False)['y2030'].max()

    return max_yield_2030_by_aez

# By scenario
max_yield_in_aez_by_sc = pd.concat([compute_max_yield_in_aez(df)
    for df in [df_current, df_uniform, df_double,
               df_equity, df_equity_and_natl_sdg, df_equity_and_village_sdg,
               df_max_growth]],
    axis=0, keys=scenario_names, names=['Scenario']).reset_index(1, drop=True).rename(
    columns={'y2030': 'maize_yield_kgha'}
).reset_index()

# By observed
max_yield_by_aez = max_yield_by_aez.rename(columns={'max_obs_yield_kgha': 'maize_yield_kgha'})
max_yield_by_aez['Scenario'] = 'Sc0: Observed'
max_yield_by_aez['level'] = 'mean'

# Merge
max_yield_in_aez_by_sc = pd.concat([
    max_yield_by_aez,
    max_yield_in_aez_by_sc
])

tbl_max_yield_by_aez = max_yield_in_aez_by_sc.pivot(index=['ZONE_AGRO_', 'level'], columns=['Scenario'], values=['maize_yield_kgha'])

# S3 Table
max_yield_by_aez = max_yield_by_aez.rename(columns={'maize_yield_kgha': 'max_obs_yield_kgha'})
tbl_scenarios_sens = pd.concat([
    pd.DataFrame([summarize_sdg_outcomes_sens(df[df.index == 'mean'])])  # run for mean predictions only
    for df in [df_current, df_uniform, df_double, df_equity_sens]],
    axis=0, keys=scenario_names[:4], names=['Scenario'])

# endregion

# region S11 Fig - Bootstrapped error and convergence of the mean from samples of residual model error in a test set

# Data - Number of maize pixels per village
# Mean annual pixels weighted by area

n_pixels_by_village = pd.merge(
    maize_village_stats_annual.assign(year=lambda df: df['year'].astype(object)),
    maize_village_stats.groupby(['Village_ID', 'year'], as_index=False, dropna=False).apply(
    lambda df: np.nan if all(df['maize_area_ha_sum'] == 0) else np.average(df['maize_pixels_n_sum'],weights=df['maize_area_ha_sum']), include_groups=False).rename(
    columns={None: 'maize_pixels_n_weighted'})
)

# Distribution in full dataset
print(n_pixels_by_village['maize_pixels_n_weighted'].describe())

# Number of villages in each quantile
print(n_pixels_by_village[
    n_pixels_by_village['yield_qntl_2019'].isin(qntl_lowest_and_highest)
].groupby('yield_qntl_2019')['Village_ID'].nunique())

# Mean number of pixels in each village by quantile
print(n_pixels_by_village[
    n_pixels_by_village['yield_qntl_2019'].isin(qntl_lowest_and_highest)
].groupby('yield_qntl_2019')['maize_pixels_n_weighted'].mean())

# Data - Bootstrapping

# Import csv
outcomes_df = pd.read_csv('data/classifier_and_yield_at_crop_data.csv')
outcomes_df = outcomes_df.replace(to_replace=-999, value=np.nan)  # clean up missing values

# Yield testing set
maize_yield_ground_truth = outcomes_df.copy().loc[
    ~np.isnan(outcomes_df['optimalClass']) &  # not in protected area
    (outcomes_df.set == "testing") &   # not included in model building
    (outcomes_df.label == 'mze') &  # maize crop cuttings
    (outcomes_df.yield_kg_ha.between(89, 10000, inclusive='neither'))  # drop outliers
    ]

maize_yield_testing_error = maize_yield_ground_truth.copy()[['crop_id', 'maizeYield', 'yield_kg_ha']]
maize_yield_testing_error['res'] = maize_yield_ground_truth['yield_kg_ha'] - maize_yield_ground_truth['maizeYield']

# Bootstrap mean error from a sample of size n=1153 (min average number of pixels in quantile)
res_np = maize_yield_testing_error['res'].to_numpy()
samples_of_error = np.random.choice(res_np, size=(1153, 1000))
samples_of_me = np.apply_along_axis(np.mean, 0, samples_of_error) / 2.5  # divide by average normalization factor

# Determine converge of mean
classifier_yields_np = maize_yield_testing_error['maizeYield'].to_numpy() / 2.5 # divide by average normalization factor
samples_of_classifier_yields = np.random.choice(classifier_yields_np, size=(1153, 1000))

def compute_running_average(a):
    running_sum = np.cumsum(a, dtype=float)
    running_avg = [x / (i+1) for i, x in enumerate(running_sum)]
    return running_avg

samples_of_classifier_yields_running_mean = np.apply_along_axis(compute_running_average, 0, samples_of_classifier_yields)
samples_of_classifier_yields_running_mean = pd.DataFrame(samples_of_classifier_yields_running_mean)
samples_of_classifier_yields_running_mean['n_samples'] = samples_of_classifier_yields_running_mean.index+1
samples_of_classifier_yields_running_mean_long = pd.melt(samples_of_classifier_yields_running_mean, id_vars='n_samples', var_name='run', value_name='running_mean')

# S11 Fig
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12, 6), dpi=300)

ax1.hist(samples_of_me, bins=25)
ax1.set_title('Bootstrapped residual mean error')
ax1.set_xlabel('Error (kg/ha)')
ax1.set_ylabel('Frequency')

g = sns.lineplot(samples_of_classifier_yields_running_mean_long, x='n_samples', y='running_mean', ax=ax2)
g.set_title('Convergence of mean yield')
g.set_xlabel('# of observations')
g.set_ylabel('Running mean (kg/ha)')

plt.show()
# endregion

# endregion