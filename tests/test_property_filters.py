import pandas as pd

from property_filters import property_types_for_bhk_and_flat


def test_filters_by_bhk_and_flat_type():
    df = pd.DataFrame({
        "BHK_Count": [3, 3, 3, 4, 3],
        "Flat Type": ["Floor", "Apartment", "Floor", "Floor", "Floor"],
        "Property Type": [
            "Independent Floor", "Apartment",
            "Builder Floor", "Independent Floor", None,
        ],
    })

    result = property_types_for_bhk_and_flat(df, 3, "Floor")
    assert result == ["Builder Floor", "Independent Floor"]


def test_no_matching_combination_returns_empty_list():
    df = pd.DataFrame({
        "BHK_Count": [3],
        "Flat Type": ["Floor"],
        "Property Type": ["Independent Floor"],
    })
    assert property_types_for_bhk_and_flat(df, 4, "Floor") == []


def test_missing_columns_returns_empty_list():
    df = pd.DataFrame({
        "BHK_Count": [3],
        "Flat Type": ["Floor"],
    })
    assert property_types_for_bhk_and_flat(df, 3, "Floor") == []


def test_none_dataset_returns_empty_list():
    assert property_types_for_bhk_and_flat(None, 3, "Floor") == []


def test_whitespace_and_case_on_flat_type():
    df = pd.DataFrame({
        "BHK_Count": [2, 2],
        "Flat Type": ["Apartment", "Floor"],
        "Property Type": ["2 BHK Apartment", "2 BHK Independent Floor"],
    })
    assert property_types_for_bhk_and_flat(df, 2, "  APARTMENT  ") == ["2 BHK Apartment"]
