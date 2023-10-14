import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from patsy import dmatrices

def run_logit_model(df: pd.DataFrame, x_label: str, count_label: str, N: int, out_str:str):
    bvalid = df[x_label].notnull() & df[count_label].notnull()
    x = df.loc[bvalid,x_label].astype(np.float64)
    p_ewe = df.loc[bvalid,count_label].astype(np.float64)
    p_none = N - df.loc[bvalid,count_label].astype(np.float64)

    tdf = pd.DataFrame(np.transpose([x,p_ewe,p_none]), columns=['val', 'count_ewe', 'count_no'])
    y_data,X_data = dmatrices(f'count_ewe + count_no ~ val', tdf)

    bin_model = sm.GLM(y_data, X_data, family=sm.families.Binomial())
    bin_fit = bin_model.fit()

    with open(f'fits/{out_str}_fit.log', 'wt') as f:
        f.write(str(bin_fit.summary()))
    return bin_fit


# def get_expected_counts(df: pd.DataFrame, val_label: str, count_label: str):
#     bvalid = df.loc[:,val_label].notnull() & df.loc[:,count_label].notnull()
#     vals = df.loc[bvalid,val_label].astype(np.float64)
#     counts = df.loc[bvalid,count_label].astype(np.float64)

#     t = pd.DataFrame(np.array([vals.values,counts.values]).transpose(), columns=['Val','Count'])
#     chk = t.groupby('Val').mean()

#     plt.scatter(chk.index,chk['Count'].values)
#     plt.show()

def med_days(in_data: pd.Series, n_days: int = 3) -> pd.Series:
    return in_data.rolling(window=n_days, center=False).mean()

def load_model(out_df: pd.DataFrame, df: pd.DataFrame, x_label: str, count_label: str, N: int, prctiles: np.array, n_days: int, x_ax: str, y_ax: str, out_str:str) -> pd.DataFrame:
    mdl_fit = run_logit_model(df, x_label, count_label, N, out_str)

    x_med = med_days(df[x_label].astype(np.float64), n_days)

    bvalid = x_med.notnull() & df[count_label].notnull()
    x = x_med[bvalid]
    y_ewe = df.loc[bvalid,count_label].astype(np.float64)
    p_max = np.max(y_ewe) / N

    pad_rat = 0.3
    psz = 10000
    val_range = np.abs(np.max(x.values) - np.min(x.values))
    pad = pad_rat*val_range
    x_vals = np.linspace(np.min(x.values)-pad, np.max(x.values)+pad, psz)
    x_pred = np.transpose([np.ones((psz,)), x_vals])
    y_pred = mdl_fit.predict(x_pred)

    # Compute values closest to desired percentiles
    chk_pctiles = np.repeat([(1.0-prctiles)*p_max], repeats=psz, axis=0)
    close_idx = np.argmin(np.abs(y_pred[:,np.newaxis] - chk_pctiles), axis=0)
    pct_vals = np.round(x_vals[close_idx], decimals=1).flatten()

    fig = plt.figure()
    ax = sns.scatterplot(x=x, y=y_ewe)
    ax.set(xlabel=x_ax, ylabel=y_ax)
    # sns.regplot(x=x, y=p_ewe, logistic=True)
    pad_mult = pad_rat / (1+2*pad_rat)
    idx_range = range(int(pad_mult*psz),int((1-pad_mult)*psz+1))
    plt.plot(x_vals[idx_range],y_pred[idx_range]*N)
    plt.savefig(f'figs/{count_label}_fit.png')
    fig.clear()

    out_vals = pd.Series(pct_vals, name=out_str)
    out_pct = pd.Series(np.round((1.0-prctiles)*p_max*N, decimals=0).astype(np.int32), name=f'{out_str}_N')
    out_df = pd.concat([out_df,out_pct,out_vals], axis=1)
    return out_df


def data_prctiles(csv_file):
    df = pd.read_csv(csv_file, dtype=object)

    N = 400
    prctiles = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)
    out_df = pd.DataFrame(np.round(prctiles*100, decimals=1), columns=['Percentile'])

    # precip_m = df['MAB_precip'].astype(np.float64)
    # precip_k = df['DAG_precip'].astype(np.float64)
    # comb_precip = pd.Series(np.nanmean(np.vstack([precip_m,precip_k]), axis=0), name='precip')

    # comb_flood_count = pd.Series(df['EW3_flood_date_count'], name='flood_count')
    # comb_dwnpr_count = pd.Series(df['EW3_dwnpr_date_count'], name='dwnpr_count')
    df.loc[df['DAG_precip'] == '67','EW3_dwnpr_date_count_k'] = float('Nan')
    comb_precip = pd.Series(np.hstack([df['MAB_precip'],df['DAG_precip']]), name='precip')
    comb_flood_count = pd.Series(np.hstack([df['EW3_flood_date_count_m'],df['EW3_flood_date_count_k']]), name='flood_count')
    comb_dwnpr_count = pd.Series(np.hstack([df['EW3_dwnpr_date_count_m'],df['EW3_dwnpr_date_count_k']]), name='dwnpr_count')

    comb_max_temp = pd.Series(np.hstack([df['MAB_max_temp'], df['DAG_max_temp']]), name='max_temp')
    comb_min_temp = pd.Series(np.hstack([df['MAB_max_temp'], df['DAG_max_temp']]), name='min_temp')

    combined_df = pd.concat([comb_precip, comb_flood_count, comb_dwnpr_count], axis=1)
    out_df = load_model(out_df, combined_df, 'precip', 'flood_count', N, prctiles, 1, 'mm/day', '# flood reports', 'flood_precip')
    out_df = load_model(out_df, combined_df, 'precip', 'dwnpr_count', N, prctiles, 1, 'mm/day', '# downpour reports', 'dwnpr_precip')


    # # # Flood/downpour Mathare
    # out_df = load_model(out_df, df, 'MAB_precip', 'EW3_flood_date_count_m', N, prctiles, 1, 'mm/day', '# flood reports', 'flood_precip_m')
    # out_df = load_model(out_df, df, 'MAB_precip', 'EW3_dwnpr_date_count_m', N, prctiles, 1, 'mm/day', '# downpour reports', 'dwnpr_precip_m')
    
    # # # Flood/downpour Kibera
    # out_df = load_model(out_df, df, 'DAG_precip', 'EW3_flood_date_count_k', N, prctiles, 1, 'mm/day', '# flood reports', 'flood_precip_k')
    # out_df = load_model(out_df, df, 'DAG_precip', 'EW3_dwnpr_date_count_k', N, prctiles, 1, 'mm/day', '# downpour reports', 'dwnpr_precip_k')

    # # Heat/Drought/Cold Mathare
    # out_df = load_model(out_df, df, 'MAB_max_temp', 'EW3_heat_date_count_m', N, prctiles, 4, 'C', '# extreme heat reports', 'heat_temp_m')
    # out_df = load_model(out_df, df, 'MAB_max_temp', 'EW3_drght_date_count_m', N, prctiles, 4, 'C', '# drought reports', 'drght_temp_m')
    # out_df = load_model(out_df, df, 'MAB_min_temp', 'EW3_cold_date_count_m', N, prctiles, 4, 'C', '# exterme cold reports', 'cold_temp_m')

    # valid_idx = np.array(np.where(df.loc[:,'EW3_heat_date_count_m'].notnull())).flatten()

    # x = np.array(df.loc[valid_idx,'MAB_max_temp']).astype(np.float64)
    # y = np.array(df.loc[valid_idx-1,'MAB_max_temp']).astype(np.float64)
    # h = np.array(df.loc[valid_idx,'EW3_heat_date_count_m'].astype(np.float64))
    # sns.scatterplot(x=x, y=y, hue=h)
    # plt.show()

    out_df.to_csv('out_prctile.csv', header=True, index=False)
    

if __name__ == '__main__':
    data_prctiles('BL-FU5Meteor_and_count.csv')
