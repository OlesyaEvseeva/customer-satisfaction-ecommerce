import seaborn as sns
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import math

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

# Plot a bar chart of value counts for a specific column.
def plot_value_counts(
    df,
    column,
    sort="count",
    top_n=None,
    bar_orientation="bar",
    figsize=(6, 4),
    tick_fontsize=8,
    color=blue,
):
    """
    Plots a bar chart of value counts for a specific column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column.
        column (str): The name of the column to analyze.
        sort (str): How to sort values — "count" (default), "index", or "none".
        top_n (int or None): Limit the number of categories displayed (default = None).
        bar_orientation (str): "bar" for vertical bars (default), "barh" for horizontal bars.
        figsize (tuple): Size of the plot (width, height). Default is (6, 4).
        tick_fontsize (int): Size of the x and y tick font. Default is 8.
    """

    result = show_value_counts(df, column, sort=sort, top_n=top_n)

    # Reverse for barh to show highest at the top
    if bar_orientation == "barh":
        result = result[::-1]

    ax = result["Count"].plot(
        kind=bar_orientation,  # type: ignore
        figsize=figsize,
        title=f"Counts for '{column}'",
        color=color,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    if bar_orientation == "bar":
        plt.xlabel(column)
        plt.ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        for i, (count, pct) in enumerate(zip(result["Count"], result["Relative (%)"])):
            ax.text(
                i,
                count + max(result["Count"]) * 0.01,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    elif bar_orientation == "barh":
        plt.ylabel(column)
        plt.xlabel("Count")
        for i, (count, pct) in enumerate(zip(result["Count"], result["Relative (%)"])):
            ax.text(
                count + max(result["Count"]) * 0.01,
                i,
                f"{pct:.1f}%",
                va="center",
                fontsize=8,
            )

    plt.grid(axis="y" if bar_orientation == "bar" else "x", linestyle="--")
    plt.tight_layout()
    plt.show()
# -----------------------------------------------------------------------------------------------------------------------------

# Plot a grid of countplots for multiple categorical columns
def plot_countplots_grid(df, columns, ncols=3, color=blue, rotation=0):
    """
    Plots countplots for a list of categorical columns in a grid layout.

    Parameters:
    - df: DataFrame
    - columns: list of column names to plot
    - ncols: number of columns in the subplot grid (default = 3)
    - color: bar color for all plots (default = blue)
    """
    n = len(columns)
    nrows = math.ceil(n / ncols)
    figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.countplot(data=df, x=col, ax=ax, color=color)

        ax.set_title(f"Count of {col.replace('_', ' ').title()}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=rotation)

        total = df[col].notna().sum()
        for p in ax.patches:
            count = p.get_height()
            percent = 100 * count / total
            ax.annotate(
                f"{percent:.1f}%",
                (p.get_x() + p.get_width() / 2, count),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Remove unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ========================================================================================================================
# REVIEW SCORE VISUALIZATION
# ========================================================================================================================
review_group_order = ["1-2 stars", "3-4 stars", "5 stars"]

# Plot AVERAGE REVIEW SCORES for a single categorical variable.
def plot_review_score_by_single_category(
    df,
    category_col,
    label_map=None,
    color=blue,
    highlight=red,
    figsize=(8, 5),
    rotation=45,
    count_limit=None,
    show_values=True,
    title=None,
):
    """
    Plot average review scores for each category in a single categorical column.

    Parameters:
    - df: DataFrame
    - category_col: the name of the categorical column
    - label_map: dict mapping internal names to display labels (optional)
    - color: bar color
    - highlight: color for overall mean line
    - figsize: figure size
    - rotation: rotation of x-tick labels
    - count_limit: filter categories with fewer than this number of rows (optional)
    - show_values: whether to annotate values on bars
    - title: chart title (optional)
    """
    overall_mean = df["review_score"].mean()

    df_filtered = df.copy()
    if count_limit is not None:
        counts = df_filtered[category_col].value_counts()
        valid_cats = counts[counts >= count_limit].index
        df_filtered = df_filtered[df_filtered[category_col].isin(valid_cats)]

    means = (
        df_filtered.groupby(category_col)["review_score"]
        .mean()
        .sort_values(ascending=False)
    )

    labels = [
        label_map.get(cat, str(cat)) if label_map else str(cat) for cat in means.index
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(labels, means.values, color=color, edgecolor="black")

    if show_values:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.03,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, 5)
    ax.set_ylabel("Average Review Score")
    ax.set_title(
        title
        if title
        else f"Average Review Score by {category_col.replace('_', ' ').title()}"
    )
    ax.tick_params(axis="x", rotation=rotation)

    # Add overall mean line
    ax.axhline(overall_mean, color=highlight, linestyle="--", linewidth=1.5)
    ax.text(
        len(labels) - 0.1,
        overall_mean + 0.02,
        f"Avg: {overall_mean:.2f}",
        color=highlight,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------------------------------------------------------------------