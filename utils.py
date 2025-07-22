import seaborn as sns
import pandas as pd
from IPython.display import display

# ========================================================================================================================
# CUSTOM COLORS & PALETTES
# ========================================================================================================================
blue = "#0A2F4F"
yellow = "#FBC02D"
red = "#B22222"
green = "#2A712D"
purple = "#5F1E7B"
blue_shades = sns.light_palette(blue, n_colors=6, reverse=True, input="hex")
color_palette = [blue, yellow, red, green, purple]
triple_palette = [red, yellow, green]
duo_palette = {
    True: yellow,
    False: blue,
}

# ========================================================================================================================
# DATA INSPECTION HELPERS
# ========================================================================================================================

# Summarize basic structure of a DataFrame(s)
def summarize_table_shapes(data):
    """
    Summarizes basic shape and structure info for one or more DataFrames.

    Parameters:
        data (dict or pd.DataFrame): A dictionary of DataFrames or a single DataFrame.

    Returns:
        pd.DataFrame: A summary table with:
            - name
            - number of rows and columns
            - list of column names
            - number of duplicate rows
    """
    if isinstance(data, pd.DataFrame):
        data = {"df": data}

    summary = []
    for name, df in data.items():
        summary.append(
            {
                "name": name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "duplicates": df.duplicated().sum(),
            }
        )
    display(pd.DataFrame(summary))
    return pd.DataFrame(summary)
# ------------------------------------------------------------------------------------------------------------------------

# Column overview of a DataFrame(s)
def column_overview(data):
    """
    Provides a detailed overview of each column in one or more DataFrames.

    Parameters:
        data (dict or pd.DataFrame): A dictionary of DataFrames or a single DataFrame.

    Displays:
        For each DataFrame:
            - dtype
            - non-null count
            - missing value count and percentage
            - number of unique values
            - list of unique values
    """
    if isinstance(data, pd.DataFrame):
        data = {"df": data}

    for name, df in data.items():
        print(f"{name.capitalize()}:")
        summary = pd.DataFrame(
            {
                "dtype": df.dtypes,
                "non_null": df.count(),
                "missing_n": df.isna().sum(),
                "missing_%": df.isna().mean() * 100,
                "uniques_n": df.nunique(),
                "uniques": [df[col].unique() for col in df.columns],
            }
        )
        display(summary)
        print("-" * 130)
# ------------------------------------------------------------------------------------------------------------------------

# Descriptive statistics for numeric columns in a DataFrame(s)
def describe_numeric(data):
    """
    Displays descriptive statistics for numeric columns in one or more DataFrames.

    Parameters:
        data (dict or pd.DataFrame): A dictionary of DataFrames or a single DataFrame.

    Notes:
        Skips any DataFrame without numeric columns.
    """
    if isinstance(data, pd.DataFrame):
        data = {"df": data}

    for name, df in data.items():
        print(f"{name.capitalize()}:")
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            print("No numeric columns to describe.")
        else:
            display(numeric_df.describe().T)

        print("-" * 130)

# ========================================================================================================================
# DATA MANAGEMENT HELPERS
# ========================================================================================================================


# Copying one or more DataFrames
def copy_dataframes(data, exclude=None):
    """
    Creates a deep copy of one or more DataFrames.

    Parameters:
        data (dict): Dictionary of DataFrames.
        exclude (list): List of keys to exclude from copying.

    Returns:
        dict: A new dictionary with copies of the DataFrames.
    """
    if not isinstance(data, dict):
        raise ValueError("copy_dataframes expects a dictionary of DataFrames.")

    exclude = exclude or []
    return {name: df.copy() for name, df in data.items() if name not in exclude}
# ------------------------------------------------------------------------------------------------------------------------

# Converting data types
def convert_column_dtypes(data, conversion_dict):
    """
    Converts column data types based on a nested or flat mapping.

    Parameters:
        data (dict or pd.DataFrame): A dictionary of DataFrames or a single DataFrame.
        conversion_dict (dict): Either a flat dictionary (for single DataFrame)
                                or a nested dictionary {table_name: {col: dtype}}.

    Notes:
        Modifies data in place.
    """
    if isinstance(data, pd.DataFrame):
        for col, dtype in conversion_dict.items():
            if col in data.columns:
                data[col] = data[col].astype(dtype)
        return

    for name, df in data.items():
        if name not in conversion_dict:
            continue

        for col, dtype in conversion_dict[name].items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

# ========================================================================================================================
# EDA HELPERS
# ========================================================================================================================


# Show a table with absolute and relative value counts for a column.
def show_value_counts(df, column, sort="count", top_n=None):
    """
    Returns a DataFrame with absolute and relative value counts
    for a specific column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column.
        column (str): The name of the column to analyze.
        sort (str): How to sort values — "count" (default), "index", or "none".
        top_n (int or None): Limit the number of categories displayed (default = None).

    Returns:
        pd.DataFrame: Table with absolute and relative counts.
    """
    abs_counts = df[column].value_counts(dropna=False)
    rel_counts = df[column].value_counts(normalize=True, dropna=False) * 100

    if sort == "count":
        abs_counts = abs_counts.sort_values(ascending=False)
    elif sort == "index":
        abs_counts = abs_counts.sort_index()

    rel_counts = rel_counts[abs_counts.index]

    result = pd.DataFrame({"Count": abs_counts, "Relative (%)": rel_counts.round(2)})

    if top_n is not None:
        result = result.head(top_n)

    return result
# ------------------------------------------------------------------------------------------------------------------------
