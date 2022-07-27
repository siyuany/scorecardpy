# -*- coding: utf-8 -*-

import pandas as pd
from pandas.api.types import CategoricalDtype
import pkg_resources


def germancredit():
    """
    German Credit Data
    ------
    Credit data that classifies debtors described by a set of 
    attributes as good or bad credit risks. See source link 
    below for detailed information.
    [source](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
    
    Params
    ------
    
    Returns
    ------
    DataFrame
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # # data structure
    # dat.shape
    # dat.dtypes
    """
    DATA_FILE = pkg_resources.resource_filename('scorecardpy',
                                                'data/germancredit.csv')

    dat = pd.read_csv(DATA_FILE)
    # categorical levels
    cate_levels = {
        "status_of_existing_checking_account": [
            '... < 0 DM', '0 <= ... < 200 DM',
            '... >= 200 DM / salary assignments for at least 1 year',
            'no checking account'
        ],
        "credit_history": [
            "no credits taken/ all credits paid back duly",
            "all credits at this bank paid back duly",
            "existing credits paid back duly till now",
            "delay in paying off in the past",
            "critical account/ other credits existing (not at this bank)"
        ],
        "savings_account_and_bonds": [
            "... < 100 DM", "100 <= ... < 500 DM", "500 <= ... < 1000 DM",
            "... >= 1000 DM", "unknown/ no savings account"
        ],
        "present_employment_since": [
            "unemployed", "... < 1 year", "1 <= ... < 4 years",
            "4 <= ... < 7 years", "... >= 7 years"
        ],
        "personal_status_and_sex": [
            "male : divorced/separated", "female : divorced/separated/married",
            "male : single", "male : married/widowed", "female : single"
        ],
        "other_debtors_or_guarantors": ["none", "co-applicant", "guarantor"],
        "property": [
            "real estate", "building society savings agreement/ life insurance",
            "car or other, not in attribute Savings account/bonds",
            "unknown / no property"
        ],
        "other_installment_plans": ["bank", "stores", "none"],
        "housing": ["rent", "own", "for free"],
        "job": [
            "unemployed/ unskilled - non-resident", "unskilled - resident",
            "skilled employee / official",
            "management/ self-employed/ highly qualified employee/ officer"
        ],
        "telephone": ["none", "yes, registered under the customers name"],
        "foreign_worker": ["yes", "no"]
    }

    def cate_type(levels):
        return CategoricalDtype(categories=levels, ordered=True)

    for i in cate_levels.keys():
        dat[i] = dat[i].astype(cate_type(cate_levels[i]))

    return dat
