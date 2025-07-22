import seaborn as sns

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

