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
