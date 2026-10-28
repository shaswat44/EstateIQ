import pandas as pd


def property_types_for_bhk_and_flat(
    df: pd.DataFrame | None,
    bhk: int | float,
    flat_type: str | None,
) -> list[str]:
    if df is None or df.empty:
        return []
    if not {"BHK_Count", "Flat Type", "Property Type"}.issubset(df.columns):
        return []
    if flat_type is None:
        return []

    try:
        bhk_num = float(bhk)
    except (TypeError, ValueError):
        return []

    selected_ft = str(flat_type).strip()
    if not selected_ft or selected_ft.lower() == "nan":
        return []

    bhk_values = pd.to_numeric(df["BHK_Count"], errors="coerce")
    flat_values = df["Flat Type"].astype("string").str.strip()

    mask = bhk_values.eq(bhk_num) & flat_values.str.casefold().eq(selected_ft.casefold())

    options = df.loc[mask, "Property Type"].dropna().astype(str).str.strip()
    return sorted(
        value for value in options.unique()
        if value and value.lower() != "nan"
    )
