import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Hamilton County Value Predictor", layout="centered")

st.title("Hamilton County Property Value Predictor")
st.write(
    "This educational app predicts **APPRAISED_VALUE** using a cleaned subset of the "
    "Hamilton County Assessor dataset."
)

# --- Robust path to the data file (works locally + on Streamlit Cloud) ---
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "cleaned_housing_data.xlsx"

@st.cache_data
def load_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Helpful debugging info in the app UI
        st.error(f"Could not find data file at: {path}")
        st.write("Files in app directory:", [p.name for p in path.parent.iterdir()])
        st.stop()

    df = pd.read_excel(path)  # no sheet_name needed for cleaned file
    df = df.dropna()
    df = df[df["APPRAISED_VALUE"] > 0]
    return df

@st.cache_resource
def train_model(df: pd.DataFrame) -> LinearRegression:
    X = df[["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]]
    y = df["APPRAISED_VALUE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

df = load_clean_data(DATA_PATH)
model = train_model(df)

st.subheader("Enter Property Features")

land_value = st.number_input("LAND_VALUE ($)", min_value=0.0, value=50000.0, step=1000.0)
build_value = st.number_input("BUILD_VALUE ($)", min_value=0.0, value=150000.0, step=1000.0)
yarditems_value = st.number_input("YARDITEMS_VALUE ($)", min_value=0.0, value=0.0, step=500.0)
calc_acres = st.number_input("CALC_ACRES (acres)", min_value=0.0, value=0.25, step=0.01)

if st.button("Predict APPRAISED_VALUE"):
    X_new = pd.DataFrame([{
        "LAND_VALUE": land_value,
        "BUILD_VALUE": build_value,
        "YARDITEMS_VALUE": yarditems_value,
        "CALC_ACRES": calc_acres
    }])
    pred = model.predict(X_new)[0]
    st.success(f"Predicted APPRAISED_VALUE: ${pred:,.0f}")

st.caption("Disclaimer: Educational use only. Not for legal, tax, or appraisal decisions.")
