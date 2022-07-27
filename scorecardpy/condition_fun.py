# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
import re
from pandas.api.types import is_numeric_dtype


def str_to_list(x):
    if x is not None and isinstance(x, str):
        x = [x]
    return x


def check_const_cols(dat):
    unique1_cols = [i for i in list(dat) if len(dat[i].unique()) == 1]
    if len(unique1_cols) > 0:
        warnings.warn(
            "There are {} columns have only one unique values, which are "
            "removed from input dataset. \n (ColumnNames: {})".format(
                len(unique1_cols), ', '.join(unique1_cols)))
        dat = dat.drop(unique1_cols, axis=1)
    return dat


def check_datetime_cols(dat):
    datetime_cols = dat.apply(pd.to_numeric, errors='ignore') \
                       .select_dtypes(object) \
                       .apply(pd.to_datetime, errors='ignore') \
                       .select_dtypes('datetime64') \
                       .columns.tolist()
    if len(datetime_cols) > 0:
        warnings.warn(
            "There are {} date/time type columns are removed from input dataset"
            ". \n (ColumnNames: {})".format(len(datetime_cols),
                                            ', '.join(datetime_cols)))
        dat = dat.drop(datetime_cols, axis=1)
    return dat


def check_cateCols_uniqueValues(dat, var_skip=None):
    char_cols = [i for i in list(dat) if not is_numeric_dtype(dat[i])]
    if var_skip is not None:
        char_cols = list(set(char_cols) - set(str_to_list(var_skip)))
    char_cols_too_many_unique = [
        i for i in char_cols if len(dat[i].unique()) >= 50
    ]
    if len(char_cols_too_many_unique) > 0:
        print('>>> There are {} variables have too many unique non-numeric '
              'values, which might cause the binning process slow. Please '
              'double check the following variables: \n{}'.format(
                  len(char_cols_too_many_unique),
                  ', '.join(char_cols_too_many_unique)))
        print('>>> Continue the binning process?')
        print('1: yes \n2: no')
        cont = int(input("Selection: "))
        while cont not in [1, 2]:
            cont = int(input("Selection: "))
        if cont == 2:
            raise SystemExit(0)
    return None


def rep_blank_na(dat):
    if dat.index.duplicated().any():
        dat = dat.reset_index(drop=True)
        warnings.warn(
            'There are duplicated index in dataset. The index has been reset.')

    blank_cols = [
        col for col in list(dat) if dat[col].astype(str).str.findall(
            r'^\s*$').apply(lambda x: 0 if len(x) == 0 else 1).sum() > 0
    ]
    if len(blank_cols) > 0:
        warnings.warn(
            'There are blank strings in {} columns, which are replaced with '
            'NaN. \n (ColumnNames: {})'.format(len(blank_cols),
                                               ', '.join(blank_cols)))
        for col in blank_cols:
            dat.loc[dat[col] == "", col] = np.nan

    # replace inf with -999
    cols_num = [col for col in list(dat) if col not in blank_cols]
    if len(cols_num) > 0:
        cols_inf = [
            col for col in cols_num
            if np.any(np.isinf(dat[col]))
        ]
        if len(cols_inf) > 0:
            warnings.warn(
                'There are infinite or NaN values in {} columns, which '
                'are replaced with -999.\n (ColumnNames: {})'.format(
                    len(cols_inf), ', '.join(cols_inf)))
            for col in cols_inf:
                dat.loc[np.isinf(dat[col]), col] = -999

    return dat


def check_y(dat, y, positive):
    positive = str(positive)
    if not isinstance(dat, pd.DataFrame):
        raise Exception("Incorrect inputs; dat should be a DataFrame.")
    elif dat.shape[1] <= 1:
        raise Exception(
            "Incorrect inputs; dat should be a DataFrame with at least two"
            " columns."
        )

    y = str_to_list(y)
    if len(y) != 1:
        raise Exception("Incorrect inputs; the length of y should be one")

    y = y[0]
    if y not in dat.columns:
        raise Exception(
            "Incorrect inputs; there is no \'{}\' column in dat.".format(y))

    if dat[y].isnull().any():
        warnings.warn(
            "There are NaNs in \'{}\' column. The rows with NaN in \'{}\' were"
            " removed from dat.".format(y, y))
        dat = dat.dropna(subset=[y])

    if is_numeric_dtype(dat[y]):
        dat.loc[:, y] = dat[y].apply(lambda x: x if pd.isnull(x) else int(x))
    unique_y = np.unique(dat[y].values)
    if len(unique_y) == 2:
        if True in [bool(re.search(positive, str(v))) for v in unique_y]:
            y1 = dat[y]
            y2 = dat[y].apply(lambda x: 1
                              if str(x) in re.split('\\|', positive) else 0)
            if np.any((y1 != y2)):
                dat.loc[:, y] = y2
                warnings.warn(
                    "The positive value in \"{}\" was replaced by 1 and "
                    "negative value by 0.".format(y))
        else:
            raise Exception(
                "Incorrect inputs; the positive value in \"{}\" is not "
                "specified" .format(y))
    else:
        raise Exception(
            "Incorrect inputs; the length of unique values in y column "
            "\'{}\' != 2.".format(y))

    return dat


def check_print_step(print_step):
    if not isinstance(print_step, (int, float)) or print_step < 0:
        warnings.warn(
            "Incorrect inputs; print_step should be a non-negative integer. "
            "It was set to 1."
        )
        print_step = 1
    return print_step


def x_variable(df, y=None, x=None, var_skip=None):
    """
    根据数据集、y变量、x变量及排除变量 (var_skip) 生成X变量列表

    - 20220727: 接口变更，入参y及x由位置参数改为关键字参数，支持默认不传入

    Parameters
    ----------
    df: pd.DataFrame 数据集
    y: （可选）Y变量名称，可以为空
    x: （可选）X变量名称列表，可以为空，默认选择df中所有列
    var_skip: （可选）需要排除的列名列表

    Returns
    -------
    list, X变量名列表
    """
    if y is None:
        excluded_cols = []
    else:
        excluded_cols = str_to_list(y)
    if var_skip is not None:
        excluded_cols = excluded_cols + str_to_list(var_skip)
    x_all = list(set(df.columns.tolist()) - set(excluded_cols))

    if x is None:
        x = x_all
    else:
        x = str_to_list(x)

        # if any([i in list(x_all) for i in x]) is False:
        if not set(x).intersection(x_all):
            x = x_all
        else:
            x_not_in_x_all = set(x).difference(x_all)
            if len(x_not_in_x_all) > 0:
                warnings.warn(
                    "Incorrect inputs; there are {} variables are not exist "
                    "in input data, which are removed from x. \n({})".format(
                        len(x_not_in_x_all), ', '.join(x_not_in_x_all)))
                x = set(x).intersection(x_all)

    return list(x)


def check_breaks_list(breaks_list, xs):
    if breaks_list is not None:
        # is string
        if isinstance(breaks_list, str):
            breaks_list = eval(breaks_list)
        # is not dict
        if not isinstance(breaks_list, dict):
            raise Exception("Incorrect inputs; breaks_list should be a dict.")
    return breaks_list


def check_special_values(special_values, xs):
    if special_values is not None:
        if isinstance(special_values, list):
            warnings.warn(
                "The special_values should be a dict. Make sure special values "
                "are exactly the same in all variables if special_values is a "
                "list."
            )
            sv_dict = {}
            for i in xs:
                sv_dict[i] = special_values
            special_values = sv_dict
        elif not isinstance(special_values, dict):
            raise Exception(
                "Incorrect inputs; special_values should be a list or dict.")
    return special_values
