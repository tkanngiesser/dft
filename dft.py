import functools
from functools import partial
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector

def wraps_partial(wrapper, *args, **kwargs):
    """ Creates a callable object whose attributes will be set from the partials nested func attribute ..."""
    wrapper = wrapper.func
    while isinstance(wrapper, functools.partial):
        wrapper = wrapper.func
    return functools.wraps(wrapper, *args, **kwargs)

class DataframeTransfomer():
    def __init__(self, df, steps):
        self.df = df
        self.steps = steps
    
    def _get_func_name(self, func):
        if isinstance(func, partial):
            name = wraps_partial(partial(func))(lambda : None).__name__
        else: 
            name = func.__name__
        return name

    def transform(self):
        for step in self.steps:
            name = self._get_func_name(step[1])
            if "cols" in name:
                if isinstance(step[2], selector):
                    cols = step[2](self.df)
                    self.df = step[1](cols=cols, df=self.df)
                else:
                    cols = step[2]
                    self.df = step[1](cols=cols, df=self.df)
            else:
                if isinstance(step[2], selector):
                    cols = step[2](self.df)
                else: 
                    cols = step[2]
                for col in cols:
                    self.df = step[1](df=self.df, col=col)
        return self.df
    
#### transform functions - more to come...

# all functions that transform > 2 cols start with prefix cols
# all functions that transform 1-n cols start with prefix col
# all function that transform 0-n vals in each of 1-n cols start with prefix val
# all specific transform function do not follow a certain naming regime
# additional parameters need to be provided by using partial

def col_lower(df, col):
    df[col] = df[col].str.lower()
    return df

def col_upper(df, col):
    df[col] = df[col].str.upper()
    return df

def col_title(df, col):
    df[col] = df[col].str.title()
    return df

def col_strip(df, col):
    df[col] = df[col].str.title()
    return df

def col_str(df, col):
    df[col] = df[col].astype(str)
    return df

def col_int(df, col):
    df[col] = df[col].astype(int)
    return df

def col_date(df, col, format):
    df[col] = pd.to_datetime(df[col], format=format)
    return df

def val_replace(df, col, val_to_replace, value):
    df[col] = df[col].replace(replace=val_to_replace, value=value)
    return df

def col_round(df, col, decimals):
    df[col] = df[col].astype(float).round(decimals)
    return df

def cols_rename(df, col_new_names):
    col_old_names = df.columns
    cols_mapping = dict(zip(col_old_names, col_new_names))
    df = df.rename(columns=cols_mapping)
    return df

def col_drop(df, col):
    df = df.dropna(subset=[col])
    return df

def col_fillna(df, col, val):
    df[col] = df[col].fillna(val)
    return df
