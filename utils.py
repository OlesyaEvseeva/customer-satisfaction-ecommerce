import seaborn as sns
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors

# ========================================================================================================================
# CUSTOM COLORS & PALETTES
# ========================================================================================================================
blue = "#0A2F4F"
yellow = "#FBC02D"
gray = "#D7CCC8"
red = "#B22222"
green = "#2A712D"
purple = "#5F1E7B"
blue_shades = sns.light_palette(blue, n_colors=6, reverse=True, input="hex")
color_palette = [blue, yellow, red, green, purple]
triple_palette = [yellow, gray, blue]
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


# ------------------------------------------------------------------------------------------------------------------------


# Visualize the distribution of a numeric column using boxplot, histogram, and KDE plot
def numplots_univariate(col, data):
    """
    Generate univariate plots for a numeric column.

    Displays three side-by-side visualizations to understand the distribution of the specified column:
    - Boxplot: identifies spread and potential outliers
    - Histogram: shows frequency distribution
    - KDE (Kernel Density Estimate): shows smoothed distribution curve

    Parameters:
    - col (str): Name of the numeric column to analyze
    - data (pd.DataFrame): DataFrame containing the column

    Returns:
    - None. Displays the plots.
    """

    fig, ax = plt.subplots(ncols=3, figsize=(16, 4))

    # Boxplot
    sns.boxplot(data=data, y=col, ax=ax[0], color=blue)
    ax[0].set_title(f"{col} - Boxplot")

    # Histogram
    sns.histplot(data[col].dropna(), bins=30, kde=False, ax=ax[1], color=blue)
    ax[1].set_title(f"{col} - Histogram")

    # KDE plot
    sns.kdeplot(data[col].dropna(), ax=ax[2], color=blue)
    ax[2].set_title(f"{col} - KDE")

    plt.tight_layout()
    plt.show()


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
def show_value_counts(df, column, sort="count", top_n=None, unique_by=None):
    """
    Returns a DataFrame with absolute and relative value counts
    for a specific column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column.
        column (str): The name of the column to analyze.
        sort (str): How to sort values — "count" (default), "index", or "none".
        top_n (int or None): Limit the number of categories displayed (default = None).
        unique_by (str or None): If provided, counts are based on unique values
                                 of this column per category instead of row counts.

    Returns:
        pd.DataFrame: Table with absolute and relative counts.
    """
    # If unique_by is set, deduplicate by that column
    if unique_by:
        df = df.drop_duplicates(unique_by)

    # Calculate absolute and relative counts
    abs_counts = df[column].value_counts(dropna=False)
    rel_counts = df[column].value_counts(normalize=True, dropna=False) * 100

    # Apply sorting
    if sort == "count":
        abs_counts = abs_counts.sort_values(ascending=False)
    elif sort == "index":
        abs_counts = abs_counts.sort_index()

    # Match order for relative counts
    rel_counts = rel_counts[abs_counts.index]

    # Build result DataFrame
    result = pd.DataFrame({"Count": abs_counts, "Relative (%)": rel_counts.round(2)})

    # Apply top_n filter if needed
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
    title=None,
    unique_by=None,
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
        color (str): Color of the bars. Default is blue.
        title (str or None): Custom title for the plot. If None, a default title is generated.
        unique_by (str or None): If provided, counts are based on unique values of this column.
    """

    result = show_value_counts(df, column, sort=sort, top_n=top_n, unique_by=unique_by)

    # Reverse for barh to show highest at the top
    if bar_orientation == "barh":
        result = result[::-1]

    # Use custom title if provided, otherwise default
    plot_title = title if title else f"Count of {column.replace('_', ' ').title()}"

    ax = result["Count"].plot(
        kind=bar_orientation,  # type: ignore
        figsize=figsize,
        title=plot_title,
        color=color,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    if bar_orientation == "bar":
        plt.xlabel(column.replace("_", " ").title())
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
        plt.ylabel(column.replace("_", " ").title())
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
def plot_countplots_grid(df, columns, ncols=3, color=blue, rotation=0, figsize=None):
    """
    Plots countplots for a list of categorical columns in a grid layout.

    Parameters:
    - df: DataFrame
    - columns: list of column names to plot
    - ncols: number of columns in the subplot grid (default = 3)
    - color: bar color for all plots (default = blue)
    - rotation: rotation angle for x-axis labels (default = 0)
    - figsize: tuple for figure size (optional).
               If None, it is calculated as (6 * ncols, 4 * nrows)
    """
    n = len(columns)
    nrows = math.ceil(n / ncols)
    if figsize is None:
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


# -----------------------------------------------------------------------------------------------------------------------------


# Annotate Brazilian states
def annotate_states(ax_list, gdf, label_col="sigla", small_states=None):
    """
    Annotates Brazilian states on one or multiple matplotlib Axes.
    Small states get offset labels with leader lines.

    Parameters
    ----------
    ax_list : list or array of matplotlib Axes
        Axes where annotations should be added.
    gdf : GeoDataFrame
        GeoDataFrame with state geometries and labels.
    label_col : str, optional
        Column in gdf containing state abbreviations (default is 'sigla').
    small_states : list, optional
        List of small state abbreviations to annotate with offset and leader lines.
    """
    if small_states is None:
        small_states = ["DF", "SE", "AL", "RJ", "ES", "RN", "PE", "PB"]

    for ax in ax_list:
        for idx, row in gdf.iterrows():
            x, y = row.geometry.centroid.coords[0]
            label = row[label_col]

            if label in small_states:
                offset_x, offset_y = x + 4.0, y - 0.3
                ax.text(
                    offset_x,
                    offset_y,
                    label,
                    fontsize=7,
                    ha="left",
                    va="center",
                    color="black",
                )
                ax.plot([x, offset_x], [y, offset_y], color="black", linewidth=0.5)
            else:
                ax.text(
                    x, y, label, ha="center", va="center", fontsize=7, color="black"
                )


# -----------------------------------------------------------------------------------------------------------------------------


# Plot absolute and log-scaled choropleth maps of Brazilian states with state labels and leader lines for small states.
def plot_state_share_maps(gdf, column, cmap="Blues", title_abs=None, title_log=None):
    """
    Plots two side-by-side choropleth maps of Brazilian states:
    one using absolute color scaling and one using logarithmic scaling.
    Both maps include state abbreviations and leader lines for small states.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing state geometries and the data column to plot.
        Must include a column 'sigla' with state abbreviations.
    column : str
        Name of the column in gdf to use for color mapping.
    cmap : str, optional, default="Blues"
        Matplotlib colormap name to use for shading the states.
    title_abs : str, optional
        Custom title for the absolute scale map. If None, a default title is used.
    title_log : str, optional
        Custom title for the log scale map. If None, a default title is used.

    Returns
    -------
    None
        Displays the two maps side by side with state labels.
    """

    # Default titles
    if title_abs is None:
        title_abs = f"Absolute Values of {column}"
    if title_log is None:
        title_log = f"{column} – Log Scale"

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # --- Absolute scale plot ---
    gdf.plot(column=column, cmap=cmap, legend=True, edgecolor="black", ax=ax[0])
    ax[0].set_title(title_abs)
    ax[0].axis("off")

    # --- Log scale plot ---
    min_val = gdf[column].replace(0, float("nan")).min() / 2  # avoid log(0)
    gdf.plot(
        column=column,
        cmap=cmap,
        legend=True,
        edgecolor="black",
        norm=mcolors.LogNorm(vmin=min_val, vmax=gdf[column].max()),
        ax=ax[1],
    )
    ax[1].set_title(title_log)
    ax[1].axis("off")

    # --- Annotate state abbreviations with leader lines for small states ---
    annotate_states(ax, gdf)

    plt.tight_layout()
    plt.show()


# ========================================================================================================================
# REVIEW SCORE VISUALIZATION
# ========================================================================================================================
review_group_order = ["Negative (1-2)", "Neutral (3)", "Positive (4-5)"]


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


# Plots AVERAGE REVIEW SCORES for one or multiple categorical variables in a grid layout.
def plot_review_score_by_categories(
    df,
    columns,
    ncols=3,
    color=blue,
    highlight=red,
    rotation=0,
    bar_orientation="bar",
    figsize=None,
    count_limit=None,
):
    """
    Plots average review scores by categorical columns in a grid layout.

    Parameters:
    - df: DataFrame
    - columns: list of column names to analyze
    - ncols: number of columns in subplot grid
    - color: bar color
    - highlight: color for overall mean line
    - rotation: x-axis label rotation
    - bar_orientation: 'bar' or 'barh'
    - figsize: overall figure size
    - count_limit: minimum number of observations to include a category (default = None = show all)
    """
    overall_mean = df["review_score"].mean()
    n = len(columns)

    # Handle 1-column case
    if n == 1:
        fig, ax = plt.subplots(figsize=figsize if figsize else (6, 4))
        axes = [ax]
        nrows = 1
        ncols = 1
    else:
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize if figsize else (6 * ncols, 4 * nrows),
        )
        axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]

        df_filtered = df.copy()

        # Filter by count_limit if specified
        if count_limit is not None:
            valid_categories = df_filtered[col].value_counts()
            valid_categories = valid_categories[valid_categories >= count_limit].index
            df_filtered = df_filtered[df_filtered[col].isin(valid_categories)]

        grouped = df_filtered.groupby(col, observed=True)["review_score"].mean()

        if bar_orientation == "barh":
            grouped = grouped.sort_values(ascending=False)

        if bar_orientation == "bar":
            sns.barplot(x=grouped.index, y=grouped.values, ax=ax, color=color)
            ax.set_title(f"Avg. Review Score by {col.replace('_', ' ')}", fontsize=10)
            ax.set_xlabel(col.replace("_", " ").title())
            ax.set_ylabel("Average Review Score")
            ax.tick_params(axis="x", rotation=rotation)
            ax.set_ylim(0, 5)

            for idx, val in enumerate(grouped.values):
                ax.text(
                    idx, val + 0.03, f"{val:.1f}", ha="center", va="bottom", fontsize=8
                )

            ax.axhline(overall_mean, color=highlight, linestyle="--", linewidth=1.5)

            min_distance = abs(grouped.values - overall_mean).min()
            if overall_mean > grouped.max():
                y_pos = overall_mean + 0.02
                x_pos = 0.95
                ha = "right"
            elif min_distance < 0.03:
                y_pos = overall_mean - 0.05
                x_pos = 0.95
                ha = "right"
            else:
                y_pos = overall_mean + 0.02
                x_pos = 0.05
                ha = "left"

            ax.text(
                x_pos,
                y_pos,
                f"Avg: {overall_mean:.1f}",
                color=highlight,
                ha=ha,
                va="bottom",
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

        elif bar_orientation == "barh":
            sns.barplot(
                y=grouped.index,
                x=grouped.values,
                ax=ax,
                color=color,
                order=grouped.index,
            )
            ax.set_title(f"Avg. Review Score by {col.replace('_', ' ')}", fontsize=10)
            ax.set_ylabel(col.replace("_", " ").title())
            ax.set_xlabel("Average Review Score")
            ax.set_xlim(0, 5)

            for idx, val in enumerate(grouped.values):
                ax.text(val + 0.03, idx, f"{val:.1f}", va="center", fontsize=8)

            ax.axvline(overall_mean, color=highlight, linestyle="--", linewidth=1.5)

            min_distance = abs(grouped.values - overall_mean).min()
            if overall_mean > grouped.max():
                x_pos = overall_mean + 0.02
                y_pos = 0.95
                va = "top"
            elif min_distance < 0.03:
                x_pos = overall_mean + 0.02
                y_pos = 0.05
                va = "bottom"
            else:
                x_pos = overall_mean + 0.02
                y_pos = 0.95
                va = "top"

            ax.text(
                x_pos,
                y_pos,
                f"Avg: {overall_mean:.1f}",
                color=highlight,
                ha="left",
                va=va,
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------


# Plot AVERAGE SHARE OF REVIEW GROUP by one or multiple categorical columns in a grid layout.
def plot_review_share_by_categories(
    df,
    review_type="negative",  # "negative", "neutral", or "positive"
    columns=None,
    ncols=3,
    color=blue,
    highlight=None,
    rotation=0,
    bar_orientation="bar",
    figsize=None,
    ymax=1.0,
    count_limit=None,
):
    """
    Plots average share of a selected review group (negative, neutral, or positive)
    by one or multiple categorical columns in a grid layout.

    Parameters:
    - df: DataFrame containing binary flag columns for review groups
          ('is_negative_review', 'is_neutral_review', 'is_positive_review')
    - review_type: 'negative', 'neutral', or 'positive' (default = 'negative')
    - columns: list of categorical columns to group by
    - ncols: number of subplot columns in the grid (default = 3)
    - color: bar color
    - highlight: color for average line (if None, it’s chosen based on review_type)
    - rotation: x-axis label rotation (only for vertical bars)
    - bar_orientation: 'bar' (vertical) or 'barh' (horizontal)
    - figsize: custom figure size
    - ymax: y-axis/x-axis maximum value (default = 1.0)
    - count_limit: minimum count per category to include (optional)
    """

    if columns is None:
        raise ValueError("Please provide a list of categorical columns to plot.")

    # --- Map review_type to column and label ---
    review_map = {
        "negative": ("is_negative_review", "Negative Reviews (1-2)", red),
        "neutral": ("is_neutral_review", "Neutral Reviews (3)", gray),
        "positive": ("is_positive_review", "Positive Reviews (4-5)", green),
    }

    if review_type not in review_map:
        raise ValueError(
            "review_type must be one of: 'negative', 'neutral', 'positive'"
        )

    flag_col, flag_label, default_highlight = review_map[review_type]
    if highlight is None:
        highlight = default_highlight

    overall_mean = df[flag_col].mean()
    n = len(columns)

    # --- Setup figure grid ---
    if n == 1:
        fig, ax = plt.subplots(figsize=figsize if figsize else (6, 4))
        axes = [ax]
    else:
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize if figsize else (6 * ncols, 4 * nrows),
        )
        axes = axes.flatten()

    # --- Loop through columns ---
    for i, col in enumerate(columns):
        ax = axes[i]
        df_filtered = df.copy()

        # Filter small categories
        if count_limit is not None:
            valid_categories = df_filtered[col].value_counts()
            valid_categories = valid_categories[valid_categories >= count_limit].index
            df_filtered = df_filtered[df_filtered[col].isin(valid_categories)]

        grouped = df_filtered.groupby(col, observed=True)[flag_col].mean()
        if bar_orientation == "barh":
            grouped = grouped.sort_values(ascending=False)

        # --- Vertical bars ---
        if bar_orientation == "bar":
            sns.barplot(x=grouped.index, y=grouped.values, ax=ax, color=color)
            ax.set_xlabel(col.replace("_", " ").title())
            ax.set_ylabel(f"Share of {flag_label}")
            ax.tick_params(axis="x", rotation=rotation)
            ax.set_ylim(0, ymax)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            # Annotate bars
            for idx, val in enumerate(grouped.values):
                ax.text(
                    idx, val + 0.005, f"{val:.1%}", ha="center", va="bottom", fontsize=8
                )

            ax.axhline(overall_mean, color=highlight, linestyle="--", linewidth=1.5)

            # Place mean annotation smartly
            min_distance = abs(grouped.values - overall_mean).min()
            if overall_mean > grouped.max():
                y_pos, ha, x_pos = overall_mean + 0.01, "right", 0.95
            elif min_distance < 0.03:
                y_pos, ha, x_pos = overall_mean - 0.05, "right", 0.95
            else:
                y_pos, ha, x_pos = overall_mean + 0.01, "left", 0.05

            ax.text(
                x_pos,
                y_pos,
                f"Avg: {overall_mean:.1%}",
                color=highlight,
                ha=ha,
                va="bottom",
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

        # --- Horizontal bars ---
        else:
            sns.barplot(
                y=grouped.index,
                x=grouped.values,
                ax=ax,
                color=color,
                errorbar=None,
                order=grouped.index,
            )
            ax.set_ylabel(col.replace("_", " ").title())
            ax.set_xlabel(f"Share of {flag_label}")
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_xlim(0, ymax)

            for idx, val in enumerate(grouped.values):
                ax.text(val + 0.005, idx, f"{val:.1%}", va="center", fontsize=8)

            ax.axvline(overall_mean, color=highlight, linestyle="--", linewidth=1.5)
            y_pos = 0.05 if overall_mean <= grouped.max() else 0.95
            ax.text(
                overall_mean + 0.01,
                y_pos,
                f"Avg: {overall_mean:.1%}",
                color=highlight,
                ha="left",
                va="bottom" if y_pos == 0.05 else "top",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

        ax.set_title(f"Share of {flag_label} by {col.replace('_', ' ')}", fontsize=10)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------


# Plot AVERAGE SHARE OF SPECIFIC-STAR REVIEWS by one or multiple categorical columns in a grid layout.
def plot_specific_star_review_share_by_categories(
    df,
    flag_col="is_5_star_review",
    columns=None,
    ncols=3,
    color=blue,
    highlight=red,
    rotation=0,
    bar_orientation="bar",
    figsize=None,
    ymax=1.0,
    count_limit=None,
):
    """
    Plots average share of a specified star rating (based on a binary flag column)
    by one or multiple categorical columns in a grid layout.

    Parameters:
    - df: DataFrame containing a binary flag column (1 if review belongs to that star group, else 0)
    - flag_col: name of the binary column (default = 'is_5_star_review')
    - columns: list of categorical columns to group by
    - ncols: number of subplot columns in the grid (default = 3)
    - color: bar color (default = blue)
    - highlight: color for average line (default = red)
    - rotation: x-axis label rotation (only for bar plots)
    - bar_orientation: 'bar' (vertical) or 'barh' (horizontal)
    - figsize: custom figure size (tuple)
    - ymax: maximum value for y-axis (or x-axis for horizontal bars)
    - count_limit: minimum count per category to include (optional)
    """

    if columns is None:
        raise ValueError("Please provide a list of categorical columns to plot.")

    # --- infer label from flag_col ---
    if "1_star" in flag_col:
        flag_label = "1-Star Reviews"
    elif "5_star" in flag_col:
        flag_label = "5-Star Reviews"
    else:
        flag_label = flag_col.replace("_", " ").title()

    overall_mean = df[flag_col].mean()
    n = len(columns)

    # --- setup figure grid ---
    if n == 1:
        fig, ax = plt.subplots(figsize=figsize if figsize else (6, 4))
        axes = [ax]
    else:
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize if figsize else (6 * ncols, 4 * nrows),
        )
        axes = axes.flatten()

    # --- plotting loop ---
    for i, col in enumerate(columns):
        ax = axes[i]
        df_filtered = df.copy()

        # Apply count filter if specified
        if count_limit is not None:
            valid_categories = df_filtered[col].value_counts()
            valid_categories = valid_categories[valid_categories >= count_limit].index
            df_filtered = df_filtered[df_filtered[col].isin(valid_categories)]

        grouped = df_filtered.groupby(col, observed=True)[flag_col].mean()

        # Sort values if plotting horizontal bars
        if bar_orientation == "barh":
            grouped = grouped.sort_values(ascending=False)

        # --- vertical bar plot ---
        if bar_orientation == "bar":
            sns.barplot(
                x=grouped.index, y=grouped.values, ax=ax, color=color, errorbar=None
            )
            ax.set_title(
                f"Share of {flag_label} by {col.replace('_', ' ')}", fontsize=10
            )
            ax.set_xlabel(col.replace("_", " ").title())
            ax.set_ylabel(f"Share of {flag_label}")
            ax.tick_params(axis="x", rotation=rotation)
            ax.set_ylim(0, ymax)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            # Annotate bars
            for idx, val in enumerate(grouped.values):
                ax.text(
                    idx, val + 0.01, f"{val:.1%}", ha="center", va="bottom", fontsize=8
                )

            # Draw average line
            ax.axhline(overall_mean, color=highlight, linestyle="--", linewidth=1.5)

            # Mean label positioning
            min_dist = abs(grouped.values - overall_mean).min()
            if overall_mean > grouped.max():
                y_pos, ha, x_pos = overall_mean + 0.01, "right", 0.95
            elif min_dist < 0.03:
                y_pos, ha, x_pos = overall_mean - 0.05, "right", 0.95
            else:
                y_pos, ha, x_pos = overall_mean + 0.01, "left", 0.05

            ax.text(
                x_pos,
                y_pos,
                f"Avg: {overall_mean:.1%}",
                color=highlight,
                ha=ha,
                va="bottom",
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

        # --- horizontal bar plot ---
        elif bar_orientation == "barh":
            sns.barplot(
                y=grouped.index,
                x=grouped.values,
                ax=ax,
                color=color,
                errorbar=None,
                order=grouped.index,
            )
            ax.set_title(
                f"Share of {flag_label} by {col.replace('_', ' ')}", fontsize=10
            )
            ax.set_ylabel(col.replace("_", " ").title())
            ax.set_xlabel(f"Share of {flag_label}")
            ax.set_xlim(0, ymax)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            # Annotate bars
            for idx, val in enumerate(grouped.values):
                ax.text(val + 0.005, idx, f"{val:.1%}", va="center", fontsize=8)

            # Draw average line
            ax.axvline(overall_mean, color=highlight, linestyle="--", linewidth=1.5)

            y_pos = 0.05 if overall_mean <= grouped.max() else 0.95
            ax.text(
                overall_mean + 0.01,
                y_pos,
                f"Avg: {overall_mean:.1%}",
                color=highlight,
                ha="left",
                va="bottom" if y_pos == 0.05 else "top",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

    # --- remove unused axes ---
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------


# Plots a GROUPED BAR CHART showing AVERAGE REVIEW SCORES for binary features (e.g., Yes/No).
# Groups are spaced out horizontally, with bars for 0 and 1 side-by-side and labeled.
# Adds an overall average line, value annotations, and custom group labels.
def plot_grouped_review_scores_binary(
    df,
    binary_cols,
    group_labels=None,
    color=blue,
    highlight=red,
    figsize=(8, 5),
    bar_width=0.45,  # wider bars
    bar_spacing=0.2,  # less space between "No" and "Yes"
    group_spacing=1.6,
    group_label_offset=-0.10,  # tighter spacing between groups
    show_values=True,
    title="Average Review Score",
):
    """
    Plot grouped (non-stacked) bar chart of average review scores by binary features.

    Parameters:
    - df: DataFrame
    - binary_cols: list of binary columns (0/1)
    - group_labels: dict mapping col names to display names
    - color: bar color
    - highlight: color for mean line + text
    - figsize: tuple
    - bar_width: width of each bar
    - bar_spacing: horizontal distance between bars in a group
    - group_spacing: horizontal distance between groups
    - show_values: whether to annotate values
    - title: title of the plot
    """
    overall_mean = df["review_score"].mean()
    fig, ax = plt.subplots(figsize=figsize)

    x_ticks = []
    x_tick_labels = []
    group_centers = []

    for idx, col in enumerate(binary_cols):
        means = df.groupby(df[col])["review_score"].mean().sort_index()  # 0 then 1
        x_center = idx * group_spacing

        xpos_no = x_center - bar_spacing / 2
        xpos_yes = x_center + bar_spacing / 2

        bars = ax.bar(
            [xpos_no, xpos_yes],
            means.values,
            width=bar_width,
            color=color,
            edgecolor="black",
        )

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

        # Tick labels under each bar
        x_ticks += [xpos_no, xpos_yes]
        x_tick_labels += ["No", "Yes"]

        # Group label centered between No and Yes
        group_label = (
            group_labels.get(col, col.replace("_", " ").title())
            if group_labels
            else col
        )
        group_centers.append((x_center, group_label))

    # Set tick labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=10)

    # Add group labels below ticks, closer to No/Yes
    for x, label in group_centers:
        ax.text(
            x,
            group_label_offset,
            label,
            ha="center",
            va="top",
            fontsize=10,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_ylim(0, 5)
    ax.set_ylabel("Average Review Score")
    ax.set_title(title)

    # Add mean line
    ax.axhline(overall_mean, color=highlight, linestyle="--", linewidth=1.5)
    ax.text(
        x_ticks[-1] + 0.4,
        overall_mean + 0.02,
        f"Avg: {overall_mean:.2f}",
        color=highlight,
        ha="right",
        va="bottom",
        fontsize=9,
        # bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------


# Plots a STACKED BAR CHART showing the distribution of REVIEW SCORE GROUPS across categories.
# Supports percentage normalization, color customization, and optional in-bar labels.
# Useful for comparing sentiment breakdown (e.g., positive/neutral/negative) by any categorical variable.
def plot_stacked_review_score_groups(
    df,
    by,
    group_col="review_score_group",
    figsize=(5, 5),
    colors=triple_palette,
    normalize=True,
    show_pct_labels=True,
    title="category",
    rotation=0,
    bar_orientation="bar",  # vertical (default) or horizontal
    count_limit=None,
    exclude_keyword="Multiple",
    sort="auto",  # "auto", "alphabetical", "negative", "positive", "neutral"
    add_average=True,  # include an Average bar
):
    """
    Plot a stacked bar chart of review_score_group by a categorical column,
    with flexible sorting and optional Average stacked bar.
    """
    # --- Filter ---
    df_plot = df.copy()
    if exclude_keyword:
        df_plot = df_plot[
            ~df_plot[by].astype(str).str.contains(exclude_keyword, case=False, na=False)
        ]
    if count_limit is not None:
        counts = df_plot[by].value_counts()
        df_plot = df_plot[df_plot[by].isin(counts[counts >= count_limit].index)]

    # --- Crosstab ---
    expected_order = ["Negative (1-2)", "Neutral (3)", "Positive (4-5)"]
    ct = pd.crosstab(df_plot[by], df_plot[group_col]).reindex(
        columns=expected_order, fill_value=0
    )
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0) * 100

    # --- Add Average row ---
    if add_average:
        global_counts = (
            df_plot[group_col].value_counts().reindex(expected_order, fill_value=0)
        )
        global_avg = (global_counts / global_counts.sum()) * (100 if normalize else 1)
        avg_row = pd.DataFrame(
            [global_avg.values], columns=expected_order, index=["AVERAGE"]
        )
        ct = pd.concat([ct, avg_row])

    # --- Sorting ---
    if sort.lower() in ["negative", "positive", "neutral"]:
        target = {
            "negative": "Negative (1-2)",
            "positive": "Positive (4-5)",
            "neutral": "Neutral (3)",
        }[sort.lower()]
        ct = ct.loc[ct[target].sort_values(ascending=False).index]

    elif sort == "alphabetical":
        ct = ct.sort_index()
        if "AVERAGE" in ct.index:
            ct = pd.concat(
                [ct.drop("AVERAGE"), ct.loc[["AVERAGE"]]]
            )  # Average always at the end

    elif sort == "auto":
        if isinstance(df[by].dtype, pd.CategoricalDtype) and df[by].dtype.ordered:
            ordered_index = df[by].dtype.categories.tolist()
            if "AVERAGE" in ct.index:
                ordered_index.append("AVERAGE")
            ct = ct.reindex(index=[i for i in ordered_index if i in ct.index])
        else:
            ct = ct.sort_index()
            if "AVERAGE" in ct.index:
                ct = pd.concat(
                    [ct.drop("AVERAGE"), ct.loc[["AVERAGE"]]]
                )  # Average last

    # --- Plot ---
    kind = "bar" if bar_orientation == "bar" else "barh"
    ax = ct.plot(
        kind=kind, stacked=True, figsize=figsize, color=colors, edgecolor="black"
    )

    # Highlight Average row with a black dashed border
    n_segments = len(expected_order)
    n_bars = len(ct)
    for i in range(n_segments):
        for j in range(n_bars):
            idx = i * n_bars + j  # patch order = segment-major
            patch = ax.patches[idx]
            if ct.index[j] == "AVERAGE":
                patch.set_edgecolor("black")  # black border
                patch.set_linewidth(1.2)  # thicker edge
                patch.set_linestyle((0, (4, 3)))  # dashed border

    # Invert y-axis for horizontal charts
    if bar_orientation == "barh":
        ax.invert_yaxis()

    # --- Labels and Grid ---
    ax.set_title(title)
    if bar_orientation == "bar":
        ax.set_xlabel("")
        ax.set_ylabel("Percentage" if normalize else "Count")
        ax.set_xticklabels(ct.index, rotation=rotation)
    else:
        ax.set_ylabel("")
        ax.set_xlabel("Percentage" if normalize else "Count")

    ax.set_axisbelow(True)
    ax.grid(axis="y" if bar_orientation == "bar" else "x", linestyle="--", alpha=0.6)

    # --- Percentage Formatter ---
    if normalize:
        (ax.yaxis if bar_orientation == "bar" else ax.xaxis).set_major_formatter(
            mtick.PercentFormatter(100)
        )

    # --- Legend ---
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(expected_order),
        frameon=False,
    )

    # --- Add % Labels ---
    if show_pct_labels and normalize:
        for j, (idx, row) in enumerate(ct.iterrows()):
            cumulative = 0
            for i, val in enumerate(row):
                if val > 3:
                    if bar_orientation == "bar":
                        ax.text(
                            j,
                            cumulative + val / 2,
                            f"{val:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if i == 2 else "black",
                        )
                    else:
                        ax.text(
                            cumulative + val / 2,
                            j,
                            f"{val:.1f}%",
                            va="center",
                            ha="center",
                            fontsize=8,
                            color="white" if i == 2 else "black",
                        )
                cumulative += val

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------


# Plots grouped stacked bar charts of review score groups for binary features.
# Each binary variable (e.g., is_heavy) is shown as a pair of "No"/"Yes" bars.
# Uses clear group labeling, normalized values, and annotated percentages.
# Ideal for comparing sentiment breakdown across multiple binary characteristics.
def plot_grouped_stacked_review_scores(
    df,
    binary_cols,
    group_col="review_score_group",
    group_labels=None,
    figsize=(10, 6),
    colors=triple_palette,
    normalize=True,
    show_pct_labels=True,
    title="Review Scores by Binary Features",
):
    expected_order = review_group_order
    bar_width = 0.35
    group_spacing = 0.4  # space between groups

    fig, ax = plt.subplots(figsize=figsize)
    x_positions = []
    bar_labels = []
    group_centers = []

    xpos = 0
    for col in binary_cols:
        ct = pd.crosstab(df[col], df[group_col])
        ct = ct.reindex(columns=expected_order, fill_value=0)

        if normalize:
            ct = ct.div(ct.sum(axis=1), axis=0) * 100

        this_group_x = []

        for val in [0, 1]:  # Explicit order: No, Yes
            cumulative = 0

            for j, grp in enumerate(expected_order):
                height = float(ct.at[val, grp])
                ax.bar(
                    xpos,
                    height,
                    bottom=cumulative,
                    width=bar_width,
                    color=colors[j],
                    edgecolor="black",
                )
                if show_pct_labels and normalize and height > 3:
                    ax.text(
                        xpos,
                        cumulative + height / 2,
                        f"{height:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if j == 2 else "black",
                    )
                cumulative += height

            x_positions.append(xpos)
            bar_labels.append("No" if val == 0 else "Yes")
            this_group_x.append(xpos)
            xpos += bar_width

        # Add center of group for the group label
        group_center = sum(this_group_x) / 2
        group_centers.append((group_center, group_labels[col] if group_labels else col))
        xpos += group_spacing  # Add spacing between groups

    # Add bar-level labels ("No"/"Yes")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, fontsize=10)

    # Add group-level labels below "No/Yes"
    # Add group-level labels below bars using data coordinates
    y_min, y_max = ax.get_ylim()
    y_label_pos = -0.1 * y_max  # slightly below zero

    for x, label in group_centers:
        ax.text(x, y_label_pos, label, ha="center", va="top", fontsize=11)

    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    if normalize:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))

    ax.legend(
        expected_order,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(expected_order),
        frameon=False,
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------
