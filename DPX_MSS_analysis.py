"""
    Dot Pattern Expectancy and Motor Selective Stop
    Task Analysis under the Dual Mechanisms of Control framework
    Author: Gordon Blake, Poldrack Lab
    Analysis Code
    TODO: Save github
"""

# DPX_FILEPATH = "data/dot_pattern_expectancy_filtered.csv"
# MSS_FILEPATH = "data/motor_selective_stop_signal_filtered.csv"

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from DPX_MSS_data_cleaning import initialize_data, trial_report

def median_split(mss_data):
    """
    Computes the median SSD for each subjects and splits their data into
    trials with SSD below median and trials with SSD above median
    (trials at the median itself are excluded)
    Returns two dataframes, mss_short, consisting of below-median trials
    for all subjects, and mss_long, consisting of above-median trials
    for all subjects
    """
    # Compute medians for all subjects
    subj_medians = mss_data.groupby(["worker_id"]).SS_delay.median()

    mss_data["delay_type"] = ""
    mss_data = mss_data.apply(label_median, args=(subj_medians,), axis=1)
    mss_short = mss_data[mss_data.delay_type == "short"]
    mss_long = mss_data[mss_data.delay_type == "long"]
    return mss_short, mss_long

def label_median(row, subj_medians):
    delay_type = "median"
    if row.SS_delay > subj_medians[row.worker_id]:
        delay_type = "long"
    elif row.SS_delay < subj_medians[row.worker_id]:
        delay_type = "short"
    row.delay_type = delay_type
    return row

def compute_dmc_measures(data):
    """
    Computes Dual Mechanisms of Control measures of proactivity and reactivity 
    for each subject as follows:
        Proactive RT = AY_RT - BY_RT
        Proactive Error = AY % Error - BY % Error
        Reactive RT = BX_RT - BY_RT
        Reactive Error = BX % Error - BY % Error
    Returns the results in a dataframe
    """

    #TODO: May want to explore normalizing these scores
    result = pd.DataFrame()
    data = trial_report(data)

    # To convert accuraracy to error rates, we subtract the accuracy from 1
    # This gives us (1 - BY_acc) - (1 - AY_acc) = AY_acc - BY_acc, so the
    # reversed order here is intentional (if confusing)
    result["proact_rt"] = data.AY_rt - data.BY_rt
    result["proact_err"] = data.BY_acc - data.AY_acc
    result["react_rt"] = data.BX_rt - data.BY_rt
    result["react_err"] = data.BY_acc - data.BX_acc

    return result

def print_pearson(title, result):
    """
    Helper function to print results of pearsonr nicely
    """
    print title
    print "r = {}, p-value = {}".format(result[0], result[1])

def within_task_comparisons(title, data):
    """
    Performs correlations of measures in data and prints the results.
    Data is a dataframe returned by compute_dmc_measures.
    Correlatons performed are:
        Proactive RT and Reactive RT
        Proactive Error and Reactive Error
        Proactive RT and Proactive Error
        Reactive RT and Reactive Error
    
    """
    print "\n" + title
    proact_vs_react_rt = pearsonr(data.proact_rt, data.react_rt)
    proact_vs_react_err = pearsonr(data.proact_err, data.react_err)
    proact_rt_vs_err = pearsonr(data.proact_rt, data.proact_err)
    react_rt_vs_err = pearsonr(data.react_rt, data.react_err)

    #TODO: Change to follow analysis plan

    print_pearson("Proactive RT vs. Reactive RT:", proact_vs_react_rt)
    print_pearson("Proactive Err vs. Reactive Err:", proact_vs_react_err)
    print_pearson("Proactive RT vs. Proactive Err:",  proact_rt_vs_err)
    print_pearson("Reactive RT vs. Reactive Err:", react_rt_vs_err)

def between_task_comparisons(dpx_dmc, mss_short_dmc, mss_long_dmc):
    dpx_mss_comparisons("DPX and MSS Short", dpx_dmc, mss_short_dmc)
    dpx_mss_comparisons("DPX and MSS Long", dpx_dmc, mss_long_dmc)



def dpx_mss_comparisons(title, dpx_dmc, mss_dmc):
    """
    Performs correlatons of the columns in dpx_dmc with the corresponding
    column in mss_dmc, which are presumed to be dataframes returned by
    compute_dmc_measures. Prints the results.
    """
    if not np.array_equal(dpx_dmc.index.values, mss_dmc.index.values): #TODO: verify that rows line up by subject, join?
        raise AssertionError("Between-task comparsion subjects not in same order.")

    for col in dpx_dmc.columns.values:
        cor = pearsonr(dpx_dmc[col], mss_dmc[col]) 
        print_pearson(title + " " + col, cor)

def compute_ssrt(data):
    pass
    # TODO: Ask Patrick for code/guidance to implement

def compare_AX_rt(dpx_data, mss_data):
    mss_data = compute_ssrt(mss_data)

    df = dpx_data.AX_rt.join(mss_data.ssrt) #Ensures that worker_ids line up
    print df
    print_pearson("MSS AX RT vs. SSRT", pearsonr(df.AX_rt, df.ssrt))


def main(): #TODO: We end up doing a lot of comparisons... ANOVA?
    # Note: to run on a different dataset, set relabel and reclean to true (or omit)
    # and dataset="discovery", "validation", or "all" Currently only one cleaned dataset
    # is cached at a time
    dpx_data, mss_data = initialize_data(relabel=False, reclean=False)
    mss_short, mss_long = median_split(mss_data)

    dpx_dmc = compute_dmc_measures(dpx_data)
    mss_short_dmc = compute_dmc_measures(mss_short)
    mss_long_dmc = compute_dmc_measures(mss_long)

    # for data in [dpx_dmc, mss_short_dmc, mss_long_dmc]: #TODO: will this work?
    #     within_task_comparisons(dpx_dmc)

    between_task_comparisons(dpx_dmc, mss_short_dmc, mss_long_dmc)


main()