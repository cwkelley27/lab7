import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------
# Data Loading + Preprocessing
# ----------------------------
@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="AssessorExport")

    # Remove missing/invalid target values
    df = df[df["APPRAISED_VALUE"].notna()]
    df = df[df["APPRAISED_VALUE"] > 0]

    # Filter to residential
    df = df[df["PROPERTY_TYPE_CODE_DESC"] == "Residential"]

    # Select predictors + target
    df = df[[
        "LAND_VALUE",
        "BUILD_VALUE",
        "YARDITEMS_VALUE",
        "CALC_ACRES",
        "APPRAISED_VALUE"
    ]].dropna()

    return df

@st.cache_resource
def train_model(df: pd.DataFrame) -> LinearRegression:
    X = df[["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]]
    y = df["APPRAISED_VALUE"]

    # Split (same idea as Task 2; inside app for simplicity)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# -------------
# Streamlit UI
# -------------
st.title("Hamilton County Property Value Predictor")
st.write(
    "This educational app predicts **APPRAISED_VALUE** using basic assessor features "
    "(land value, building value, yard items value, and acreage)."
)

st.subheader("Enter Property Features")

land_value = st.number_input("LAND_VALUE ($)", min_value=0.0, value=50000.0, step=1000.0)
build_value = st.number_input("BUILD_VALUE ($)", min_value=0.0, value=150000.0, step=1000.0)
yarditems_value = st.number_input("YARDITEMS_VALUE ($)", min_value=0.0, value=0.0, step=500.0)
calc_acres = st.number_input("CALC_ACRES (acres)", min_value=0.0, value=0.25, step=0.01)

# Load data + train model
data_path = "Housing_Hamilton_County.xlsx"
df = load_and_clean_data(data_path)
model = train_model(df)

# Predict button
if st.button("Predict APPRAISED_VALUE"):
    X_new = pd.DataFrame([{
        "LAND_VALUE": land_value,
        "BUILD_VALUE": build_value,
        "YARDITEMS_VALUE": yarditems_value,
        "CALC_ACRES": calc_acres
    }])

    pred = model.predict(X_new)[0]
    st.success(f"Predicted APPRAISED_VALUE: ${pred:,.0f}")

st.caption("Disclaimer: Educational use only. Predictions are approximate and not for appraisal or legal decisions.")
