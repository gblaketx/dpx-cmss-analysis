"""
    Dot Pattern Expectancy and Motor Selective Stop
    Task Analysis under the Dual Mechanisms of Control framework
    Author: Gordon Blake, Poldrack Lab
    Code to label and clean data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import binom, chi2_contingency
from warnings import warn

SUBJECT_ASSIGNMENT_FILEPATH = "data/subject_assignment.csv"
DPX_FILEPATH = "data/dot_pattern_expectancy.csv"
MSS_FILEPATH = "data/motor_selective_stop_signal.csv"
DPX_LABELED_FILEPATH = "data/dot_pattern_expectancy_labeled.csv"
MSS_LABELED_FILEPATH = "data/motor_selective_stop_signal_labeled.csv"
DPX_CLEANED_FILEPATH = "data/dot_pattern_expectancy_filtered.csv"
MSS_CLEANED_FILEPATH = "data/motor_selective_stop_signal_filtered.csv"

# Uncomment to store reasons for exclusion of each worker in this dataframe
# why_excluded = pd.DataFrame(columns=["worker_id", "exclusion_reason"])


def initialize_data(relabel=True, reclean=True, dataset=None):
    """
    Initializes the data sets by reading them in, cleaning, and labeling them.
    Returns two pandas dataframes with the DPX and MSS data, respectively.
    Cleaned data and labeled data are cached in CSVs denoted with the CLEANED_FILEPATH
    and LABELED_FILEPATH constants.

    relabel: If relabel is true, raw, unlabeled data is loaded from FILEPATH, labeled, and cleaned
        before being returned. The labeled data is saved to LABELED_FILEPATH during this process
    reclean :If reclean is true (and relabel is false), labeled data is loaded from LABELED_FILEPATH,
        cleaned, saved to CLEANED_FILEPATH, and returned
    
        If relabel and reclean are both false, then data is loaded from CLEANED_FILEPATH, which
        is assumed to already be cleaned and labeled. No additional pprocessing takes place,
        and the data are returned.

    dataset: string identifying which dataset to use: "discovery", "validation", or "all". Only
        subjects from this dataset will be included in the analysis. (See preclean_data)
        Note that subjects are filtered by dataset BEFORE cleaning, so this parameter will
        only have an effect if relabel is True. When loading from labeled or cleaned data,
        the function should be invoked without this parameter.
    """
    dpx_path, mss_path = (DPX_CLEANED_FILEPATH, MSS_CLEANED_FILEPATH)
    if dataset != None and not relabel:
        warn("Dataset specified. Ignoring cache. Recleaning data.")
        relabel = True
        reclean = True

    if reclean: dpx_path, mss_path = (DPX_LABELED_FILEPATH, MSS_LABELED_FILEPATH)
    if relabel: 
        if not reclean:
            reclean = True
            warn("Must reclean data after relabeling. Setting reclean to true.")
        dpx_path, mss_path = (DPX_FILEPATH, MSS_FILEPATH)

    dpx_data, mss_data = read_data(dpx_path, mss_path)

    if relabel:
        dpx_data = preclean_data(dpx_data, dataset)
        mss_data = preclean_data(mss_data, dataset)

        # Data must be labeled before it is cleaned
        dpx_data, mss_data = label_data(dpx_data, mss_data)

        dpx_data.to_csv(DPX_LABELED_FILEPATH)
        mss_data.to_csv(MSS_LABELED_FILEPATH)

    dpx_data.AXCPT_trial_type = dpx_data.AXCPT_trial_type.astype('category')
    mss_data.AXCPT_trial_type = mss_data.AXCPT_trial_type.astype('category')

    if reclean:
        dpx_data, mss_data = clean_data(dpx_data, mss_data)

        dpx_data.to_csv(DPX_CLEANED_FILEPATH)
        mss_data.to_csv(MSS_CLEANED_FILEPATH)

    return dpx_data, mss_data

def read_data(dpx_fpath, mss_fpath):
    """
    Reads DPX and MSS data into dataframe from the CSVs at the given filepaths.
    Returns a tuple containing the DPX and MSS dataframes, in that order.
    """
    dpx_data = pd.read_csv(dpx_fpath)
    assert (len(dpx_data) > 0), "Error reading DPX data"
    mss_data = pd.read_csv(mss_fpath)
    assert (len(mss_data) > 0), "Error reading MSS data"
    return dpx_data, mss_data


def preclean_data(data, dataset):
    """
    Removes practice sessions from data and includes only subjects in
    the relevant dataset.
    data: dataframe (DPX or MSS data) to be precleaned
    dataset: name of dataset to use: "discovery", "validation", or "all". Only
        subjects in the given dataset will be included in the dataframe returned
    returns dataframe with only test trials and subjects from the given dataset.
    """
    data = data[data.exp_stage == "test"]

    subj_assignments = pd.read_csv(SUBJECT_ASSIGNMENT_FILEPATH)
    assert (len(subj_assignments) > 0), "Error reading subject assignments."

    if dataset == "discovery" or dataset == "validation":
        subs = subj_assignments[subj_assignments.dataset == dataset].worker_id
    else:
        assert (dataset == "all"), "Dataset parameter has unrecognized value."
        return data

    data = data[data.worker_id.isin(subs)]

    return data


def filter_mss(mss_data, iterative=False):
    """
    Takes MSS dataframe and applies several filters to it, returning a list of subjects
    to exclude because they failed one or more check.
    If iterative is True, uses an iterative filtering algorithm (see filter_iterative_all)
    Otherwise uses a non-iterative algorithm (see filter_subjects).
    """
    excluded_mss_binomial = filter_binomial_test(mss_data)
    excluded_mss_ignore = filter_stop_vs_ignore(mss_data, excluded_mss_binomial)
    
    if iterative: 
        excluded_mss_other = filter_iterative_all("mss", mss_data)
    else:
        excluded_mss_other = filter_subjects("mss", mss_data)

    return np.unique(np.concatenate([excluded_mss_ignore, excluded_mss_binomial, excluded_mss_other]))

def filter_subjects(title, data):
    """
    Takes a labeled DPX or MSS dataframe and a title (either "dpx" or "mss") used for
    labeling and logic.
    Returns an ndarray of subjects to exclude from the data, who deviated by more than
    3 SDs from the mean subject RT or accuracy on any trial type.
    If title is "mss", then AX trials are excluded from the filtering process, as
    subject's accuracy on those trials is filtered via a binomial test, 
    and subjects do not respond on successful AX trials.
    """

    print "\nAccuracy Thresholds"
    subj_acc = data.groupby(["worker_id", "AXCPT_trial_type"]).correct_aggregate.mean().unstack("AXCPT_trial_type")
    acc_mean = subj_acc.mean()
    acc_sd = subj_acc.std()
    excluded_acc = filter_custom(subj_acc, compute_thresholds(title, '<>', acc_mean, 3, acc_sd), title  + "_acc")

    print "\nRT Thresholds"
    # Only include correct trials when computing RT means and SDs
    data_correct = data[data.correct == 1]
    subj_rt = data_correct.groupby(["worker_id", "AXCPT_trial_type"]).rt.mean().unstack("AXCPT_trial_type")
    rt_mean = subj_rt.mean()
    rt_sd = subj_rt.std()

    excluded_rt = filter_custom(subj_rt, compute_thresholds(title, '<>', rt_mean, 3, rt_sd), title  + "_rt")

    return np.unique(np.append(excluded_acc, excluded_rt))

def trial_report(data):
    """
    Given either DPX or MSS data, returns a dataframe with each subject's mean
    accuracy and response time on each trial type, (for a total of 8 columns)
    """
    report = data[["worker_id", "AXCPT_trial_type", "correct_aggregate", "rt"]]
    report_accuracy = report.groupby(["worker_id", "AXCPT_trial_type"]).correct_aggregate.mean().unstack("AXCPT_trial_type")
    report_rt = report[report.correct_aggregate == 1].groupby(["worker_id", "AXCPT_trial_type"]).rt.mean().unstack("AXCPT_trial_type")
    report = report_accuracy.join(report_rt, lsuffix="_acc", rsuffix="_rt")
    return report

def filter_iterative_all(title, data):
    """
    Iteratively filters data by the categories in trial type by selecting data for
    a single trial type and grouping it by worker_id. Calls filter_iterative, which drops
    subjects one at a time from the given data until all subjects fall within DEV_FROM_MEAN SDs of the
    subject mean. For MSS data (title is "mss"), excludes AX trials from filtering.
    (Can't filter by RT becausebsubject shouldn't respond, and 
    accuracy is filtered via the binomial test).
    """

    excluded = []
    trial_types = data.AXCPT_trial_type.cat.categories
    if title == "mss": trial_types = trial_types.drop("AX")
    print trial_types

    ids = ['s043', 's247', 's299', 's309', 's416']
    test = data[data.AXCPT_trial_type == 'BX'].groupby("worker_id").rt.mean()
    print test.filter(items=ids)

    for trial_type in trial_types:
        print "\nACC", trial_type, 
        trial_data = data[data.AXCPT_trial_type == trial_type]
        acc_data = trial_data.groupby(["worker_id"]).correct_aggregate.mean()
        excluded += filter_iterative(acc_data)
        
        print "\nRT", trial_type,
        # Only include correct trials when computing RT means and SDs
        trial_data_correct = trial_data[trial_data.correct == 1]
        rt_data = trial_data_correct.groupby(["worker_id"]).rt.mean()
        excluded += filter_iterative(rt_data)

    print ""
    return np.unique(excluded)

def filter_iterative(data):
    """
    Helper function for filter_iterative_all.
    data: Series of subject means, which are removed one at a time 
        until all entries lie within three standard deviations of the mean.
    returns a list of excluded subjects.
    """
    excluded = []
    DEV_FROM_MEAN = 3.0
    print "ITERATIVE FILTERING"
    while(True):
        mean = data.mean()
        sd = data.std()
        deviations = abs((data - mean) / sd)
        max_idx = deviations.idxmax()
        if(deviations[max_idx] < DEV_FROM_MEAN): break
        data.drop(max_idx, inplace=True)
        excluded.append(max_idx)

    print sorted(excluded)
    return excluded

def compute_thresholds(title, cmps, means, factors, sds):
    """
    Generates a threshold dataframe that can be passed to filter_custom.
    title: The name of the data being filtered. If title is mss, 'AX' is excluded from
        the filtering threshold, bexause subjects are filtered by a binomial test on AX
        trials (see filter_binomial_test).
    cmp: A symbol or list of symbols that determines how the threshold is compared with the data. 
        > means that a subject if excluded if they have a value greater than the valmax
        < means they're excluded if they have a value less than the valmin
        <> is a two-tailed exclusion where the subject is excluded if they have a value greater than
            valmax or less than valmin.
    means: A Series of means labeled by trial type
    sds: A Series of standard deviations labeled by trial type
        factors A single number or list of numbers multiplied by standard deviation to compute
        a threshold

    returns a theshold dataframe consisting of three columns.
    valmax contains the upper threshold value for each trial type, computed as means + factors * sds. 
    valmin contains the lower threshold value for each trial type, computed as mans - factors * sds.
    cmp contains the compartor to use for each trial type.
    """
    if title == 'mss': # Exclude AX trials from this threshold, as a binomial test is used
        means = means.drop('AX')
        sds = sds.drop('AX')

    # If single operator provided for comparison, convert to list of same operator
    if type(cmps) == str:
        cmps = [cmps] * len(means)

    # If single number provided, expand into list
    if type(factors) != list:
        factors = [factors] * len(means)

    if len(factors) != len(sds) or len(means) != len(sds):
        raise Exception('Arguments of improper length')

    thresh = pd.DataFrame({"valmin": means - factors*sds, "valmax": means + factors*sds})
    thresh['cmp'] = cmps

    return thresh


def filter_custom(df, thresh, title):
    """
    Filters df according to parameters in thresh

    df: dataframe containing data to filter containing data to filter.
    thresh: dataframe specifying filtering parameters.
        The row names of thresh are the name of the columns of df on which to filter
        tresh contains columns valmin and valmax with the values that serve as the filter thresholds
        and a column "cmp" with the comparison operator to use (<, >, or <>), where < excludes subjects
        below valmin, > excludes subjects below valmax, and <> excludes subjects less than valmin or
        greater than valmax. A subject for which this comparison evaluates to True will be added to 
        the list of exclusions.
    title: String to label columns in why_excluded

    Returns subjects excluded by any threshold 
    """

    to_exclude = []
    # Uncomment to store reasons for subject exclusion in this dataframe
    # global why_excluded

    print thresh

    for trial_type, row in thresh.iterrows():
        if row.cmp == '<':
            mask = df[trial_type] < row.valmin
        elif row.cmp == '>':
            mask = df[trial_type] > row.valmax
        elif row.cmp == '<>': 
            mask = np.logical_or(df[trial_type] < row.valmin, df[trial_type] > row.valmax)
        else:
            raise Exception('Invalid comparison operator')

        failed_trial_subjs = df[mask].index.values
        to_exclude = np.concatenate([to_exclude, failed_trial_subjs])

        # Uncomment to store reasons for subject exclusion
        # labels = [title + "_" + trial_type] * len(failed_trial_subjs)
        # why_excluded = why_excluded.append(pd.DataFrame({"worker_id" : failed_trial_subjs, "exclusion_reason" : labels}))

    return np.unique(to_exclude)

def filter_binomial_test(mss_data):
    """ 
    Takes MSS dataframe as argument.
    Returns a list of subjects to be excluded because they failed a
    two-tailed binomial test on the probability of success on
    critical hand stop (AX) trials in the MSS task.
    Failure is defined as a p-value of less than 0.05
    """
    # Uncomment to store reasons for subject exclusion
    # global why_excluded

    subjs_AX_correct = mss_data.groupby(["worker_id","AXCPT_trial_type"]).correct_aggregate.value_counts().unstack("AXCPT_trial_type", 0).AX.unstack("correct_aggregate", 0)
    to_exclude = subjs_AX_correct[subjs_AX_correct.apply(lambda x: sp.stats.binom_test(x) < 0.05, axis=1)]
    labels = ["mss_AX_bin"] * len(to_exclude)
    
    # Uncomment to store reasons for subject exclusion
    # why_excluded = why_excluded.append(pd.DataFrame({"worker_id" : to_exclude.index.values, "exclusion_reason" : labels}))

    return to_exclude.index.values

def filter_stop_vs_ignore(df, excluded_mss_binomial):
    """
    Adapted from Ian Eisenberg's code

    Filters the given MSS dataframe by identifying subjects who don't stop on ignore trials significantly
    less than on stop trials. (Using a chi-squared test, p < 0.05)

    df A dataframe with MSS data
    excluded_mss_binomial An nparray of workers excluded by filter_binomial_test
    
    returns An nparray if worker_ids who failed the check.
    """
    workers = np.setdiff1d(np.unique(df.worker_id), excluded_mss_binomial, assume_unique=True)
    failed_subjs = []

    stop_counts = df.query('exp_stage == "test" and condition == "stop"').groupby('worker_id').stopped.sum()
    stop_counts = df.query('condition != "go" and exp_stage == "test"').groupby(['worker_id','condition']).stopped.sum().reset_index()
    stop_counts.loc[:,'goed'] = 60 - stop_counts.stopped 

    for worker in workers:
        stop_counts[stop_counts.worker_id == worker]
        obs = np.matrix(stop_counts[stop_counts.worker_id == worker][['stopped','goed']])
        csc = chi2_contingency(obs) #TODO: Remove or fix
        p = csc[2]
        print "chi2 [1] = {}, chi2 [2] = {}".format(csc[1], csc[2])
        # p = chi2_contingency(obs)[2]
        if obs[0,0]>obs[0,1] or p<.05:
            failed_subjs.append(worker)

    return np.array(failed_subjs)   

def clean_data(dpx_data, mss_data):
    """
    Given labeled DPX and MSS dataframes, filters them by dropping subjects who fail
    any of several checks (see filter_mss and filter_subjects for details).
    Returns DPX and MSS dataframes with those subjects removed.
    Prints information about the filtering process to the console.
    """
    print "\nDPX Filtering"
    excluded_dpx = filter_subjects("dpx", dpx_data)

    print "\nMSS Filtering"
    excluded_mss = filter_mss(mss_data)

    excluded_subjs = np.unique(np.concatenate([excluded_dpx, excluded_mss]))
    print "\nExcluded ({}):\n".format(len(excluded_subjs)), excluded_subjs

    dpx_data_filtered = dpx_data[~dpx_data.worker_id.isin(excluded_subjs)]
    mss_data_filtered = mss_data[~mss_data.worker_id.isin(excluded_subjs)]

    return dpx_data_filtered, mss_data_filtered

#####################################################################################
# Labeling Functions
# Used to label dataframes before cleaning, including labeling MSS trials as AX, AY, BX, BY,
# creating two correctnes columns, one that aggregates omission and comission errors,
# and one that disaggregates them.
# Invoked by initialize_data when relabel is True.

def label_data(dpx_data, mss_data):
    """
    Relabels "correct" columns in the data
    The correct_aggregate column aggregates omission and commission errors, so 
    a row is 1 if correct and 0 otherwise
    The correct column disaggregates these errors with the following encodings:
      1: Correct response
      0: commission error
      -1: Omission error
    """
    dpx_data.rename(columns = {"condition" : "AXCPT_trial_type"}, inplace=True)

    # Label Trial Types in MSS Data (AX, BY, etc.)
    mss_data["AXCPT_trial_type"] = ""
    mss_data = mss_data.apply(label_row_trial_type, axis=1)

    dpx_data["correct_aggregate"] = ""
    mss_data["correct_aggregate"] = ""
    dpx_data.correct_aggregate = dpx_data.correct.map({1: 1, 0: 0, -1: 0}) # Replaces -1 with 0
    mss_data = mss_data.apply(label_mss_correct, axis=1)

    return dpx_data, mss_data


def label_mss_correct(row): #Note: Assumes only conditions are stop, go, and ignore
    """ 
    Relabels the "correct" column in the MSS data so the result is:
        1 if condition is stop and stopped is true
            or condition is go or ignore and key_press equals correct_response
        0 if condition is stopped and stopped is false
            or condition is go or ignore and key_press is not equal to correct_response 
    """
    if row.condition == "stop":
        row.correct_aggregate = int(row.stopped)
        row.correct = int(row.stopped)
    else: # Condition is go or ignore
        row.correct_aggregate = int(row.key_press == row.correct_response)
        if(row.key_press == -1):
            row.correct = -1
        else:
            row.correct = int(row.key_press == row.correct_response)
    return row

def label_row_trial_type(row):
    """
    Given a trial row in the MSS data, labels the AXCPT_trial_type 
    attribute as follows:
        AX: Critical hand stop trial
        AY: Critical Hand go trial
        BX: Noncritical hand ignore trial
        BY: Noncritical hand go trial
    """
    # stop_response is the key for the critical hand
    if row.correct_response == row.stop_response:
        cue = "A"
    else:
        cue = "B"

    if row.condition == "go":
        target = "Y"
    else: # Stop or ignore
        target = "X"
    
    row["AXCPT_trial_type"] = cue + target
    return row

#####################################################################################
# Exclusion Analysis:
# Functions used to compare subjects excluded by filter_mss to the set of subjects
# excldued in Ian's analysis code.
def compare_excluded(my_excluded):
    ian_excluded = pd.read_csv("data/exclusion_analysis/failed_subjects_ian.csv")
    ian_excluded = ian_excluded[ian_excluded["motor_selective_stop_signal.SSRT"] == True]

    ian_set = frozenset(ian_excluded.worker_id)
    my_set = frozenset(my_excluded)
    my_unique = my_set.difference(ian_set)
    print "Excluded in mine only ({}):\n".format(len(my_unique)), my_unique

    ian_unique = ian_set.difference(my_set)
    print "Excluded in Ian only ({}):\n".format(len(ian_unique)), ian_unique

    return my_set, ian_set

def exclusion_analysis(mss_data, excluded_mss_only):
    global why_excluded

    # my_excluded, ian_excluded = compare_excluded(excluded_mss_only)

    # mss_data_correct = mss_data[mss_data.correct == 1]
    # subj_acc = mss_data.groupby(["worker_id", "AXCPT_trial_type"]).correct_aggregate.mean().unstack("AXCPT_trial_type")
    # subj_rt =  mss_data_correct.groupby(["worker_id", "AXCPT_trial_type"]).rt.mean().unstack("AXCPT_trial_type")

    # subject_means = subj_acc.join(subj_rt, lsuffix="_acc", rsuffix="_rt")
    # subject_means["ian_excluded"] = [sub in ian_excluded for sub in subject_means.index.values]
    # subject_means["my_excluded"] = [sub in my_excluded for sub in subject_means.index.values]
    # print subject_means

    # subject_means.to_csv("data/exclusion_analysis/excluded_subjects.csv")

    exc = why_excluded.groupby("worker_id").exclusion_reason.value_counts()
    exc[exc > 0] = 1
    exc = exc.unstack("exclusion_reason")
    # exc["ian_excluded"] = [sub in ian_excluded for sub in exc.index.values]
    exc.to_csv("data/exclusion_analysis/excluded.csv")

    # print dpx_data[[id in my_unique for id in dpx_data.worker_id]]
    # print mss_data[[id in my_unique for id in mss_data.worker_id]]


def exclusion_analysis_it(mss_data, ex_bin, ex_ignore, ex_single, ex_it):
    """
    Takes as arguments the mss dataframe and several np.arrays:
        ex_bin: list of unique subjects excluded by binominal test
        ex_ignore: list of unique subjects excluded by stop_vs_ignore
        ex_single: list of unique subjects exluded by single pass filter (filter_subjects)
        ex_it: list of unique subjects excluded by iterative filter
    """
    excluded_all_single =  np.unique(np.concatenate([ex_ignore, ex_bin, ex_single]))
    excluded_all_it = np.unique(np.concatenate([ex_ignore, ex_bin, ex_it]))
    
    it_only = np.setdiff1d(ex_it, excluded_all_single, assume_unique=True)
    print "Excluded in iterative only", it_only

    single_only = np.setdiff1d(ex_single, excluded_all_it, assume_unique=True)
    print "Excluded in single-pass only", single_only

    report = mss_data[["worker_id", "AXCPT_trial_type", "correct_aggregate", "rt"]].query("worker_id in @it_only")
    report_accuracy = report.groupby(["worker_id", "AXCPT_trial_type"]).correct_aggregate.mean().unstack("AXCPT_trial_type")
    report_rt = report[report.correct_aggregate == 1].groupby(["worker_id", "AXCPT_trial_type"]).rt.mean().unstack("AXCPT_trial_type")
    report = report_accuracy.join(report_rt, lsuffix="_acc", rsuffix="_rt")
    print report
    report.to_csv("data/exclusion_analysis/excluded_it.csv")

#####################################################################################

##################################################################################### 
# Plotting Functions
# These are used only to visualize the results of cleaning, and are not invoked in the
# data labeling or cleaning process itself.

def plot_errors(title, data):
    """
    Creates histograms of the distribution of subjects with varying numbers of 
    correct responses, errors of omission and comission, by trial type.
    """
    errs = data.groupby(["AXCPT_trial_type"]).correct.value_counts(normalize=True)

    print errs
    print "Omission", errs.iloc[errs.index.get_level_values('correct') == -1]
    print "commission", errs.iloc[errs.index.get_level_values('correct') == 0]

    subj_errs = data.groupby(["worker_id", "AXCPT_trial_type"]).correct.value_counts(normalize=True).unstack(["AXCPT_trial_type", "correct"], 0).fillna(0).stack("correct").fillna(0)
    print subj_errs

    plot_hists(title + "_omission", subj_errs.iloc[subj_errs.index.get_level_values('correct') == -1])
    plot_hists(title + "_commission", subj_errs.iloc[subj_errs.index.get_level_values('correct') == 0])
    plot_hists(title + "_correct", subj_errs.iloc[subj_errs.index.get_level_values('correct') == 1])

def plot_hists(title, df):
    """
    Plots and saves a histogram of df performance, with the 4 trial types plotted side by side.
    """
    print "Subj count " + title, len(df['AX'])
    savename = 'plots/'+ title + '.png'
    plt.hist([df['AX'], df['AY'], df['BX'], df['BY']], label=['AX', 'AY', 'BX', 'BY'], 
        bins=[0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #Note: histogram 9 bins to capture BY (only 8 trials)

    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(savename)
    plt.clf()

def plot_subject_means(title, data):
    """
    Plots histograms of mean subject RT and accuracy on each of the four trial types
    (except RT for AX trials, which is excluded from MSS data).
    """
    report = trial_report(data)

    fig = plt.figure()

    for idx, col in enumerate(report.columns.values):
        if col == "AX_rt": continue
        subplot = fig.add_subplot(2, 4, idx + 1)
        subplot.hist(report[col])
        subplot.set_title(col)

    plt.savefig("plots/" + title + ".png")
    plt.clf()

def plot_filter_compare(title, l1, l2, data_before, data_after):
    """
    Plots histograms of data_before and data_after on top of each other,
    labeled with l1 and l2, respectively.
    Used to compare data before and after filtering, or the effect of
    two different filtering algorithms.
    """
    report_before = trial_report(data_before)
    report_after = trial_report(data_after)

    fig = plt.figure()
    # plt.title(title)
    subplots = []
    for idx, col in enumerate(report_before.columns.values):
        if col == "AX_rt": continue

        subplot = fig.add_subplot(2, 4, idx + 1)
        subplots.append(subplot)
        # subplot.hist(report[col])
        # f, ax = plt.subplots()
        bins = np.linspace(report_before[col].min(), report_before[col].max(), 35)

        sns.distplot(report_before[col], ax=subplot, kde=False, bins=bins, label=l1)
        # bins = np.histogram(report_before[col])[1]
        sns.distplot(report_after[col], ax=subplot, kde=False, bins=bins, label=l2)


    
    plt.legend(loc='best')
    plt.savefig("plots/" + title + ".png")
    plt.clf()

    for idx, col in enumerate(report_before.columns.values):
        if col == "AX_rt": continue
        fig, ax = plt.subplots()
        bins = np.linspace(report_before[col].min(), report_before[col].max(), 35)

        sns.distplot(report_before[col], ax=ax, kde=False, bins=bins, label=l1)
        # bins = np.histogram(report_before[col])[1]
        sns.distplot(report_after[col], ax=ax, kde=False, bins=bins, label=l2)
        plt.legend(loc='best')
        plt.savefig("plots/" + title + "_" + col + ".png")
        plt.clf()

#####################################################################################   

def main():
    """
    Invoke main to label and clean the data and save it.
    """
    dpx_data, mss_data = initialize_data(relabel=False, reclean=False, dataset="discovery")


# Uncomment to run entire labeling and cleaning routine