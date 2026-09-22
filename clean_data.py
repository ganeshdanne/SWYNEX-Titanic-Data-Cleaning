import pandas as pd
import os

# ==========================================
# 1. CREATE FOLDERS
# ==========================================

os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ==========================================
# 2. LOAD PUBLIC DATASET
# ==========================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("Dataset loaded successfully!")


# ==========================================
# 3. SAVE ORIGINAL DATASET
# ==========================================

df.to_csv("data/titanic_raw.csv", index=False)


# ==========================================
# 4. DISPLAY ORIGINAL INFORMATION
# ==========================================

print("\n========== ORIGINAL DATA ==========")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# ==========================================
# 5. STANDARDIZE COLUMN NAMES
# ==========================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nStandardized column names:")
print(df.columns.tolist())


# ==========================================
# 6. REMOVE DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

df = df.drop_duplicates()

print("\nDuplicates removed:", duplicates)


# ==========================================
# 7. CLEAN TEXT VALUES
# ==========================================

if "sex" in df.columns:
    df["sex"] = (
        df["sex"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

if "embarked" in df.columns:
    df["embarked"] = (
        df["embarked"]
        .astype("string")
        .str.strip()
        .str.upper()
    )


# ==========================================
# 8. CONVERT NUMERICAL COLUMNS
# ==========================================

numeric_columns = [
    "passengerid",
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ==========================================
# 9. CHECK INVALID AGE
# ==========================================

invalid_age = 0

if "age" in df.columns:

    invalid_age_condition = (
        (df["age"] < 0) |
        (df["age"] > 100)
    )

    invalid_age = invalid_age_condition.sum()

    df.loc[
        invalid_age_condition,
        "age"
    ] = pd.NA


# ==========================================
# 10. CHECK INVALID FARE
# ==========================================

invalid_fare = 0

if "fare" in df.columns:

    invalid_fare_condition = df["fare"] < 0

    invalid_fare = invalid_fare_condition.sum()

    df.loc[
        invalid_fare_condition,
        "fare"
    ] = pd.NA


# ==========================================
# 11. HANDLE MISSING AGE
# ==========================================

age_missing_before = 0

if "age" in df.columns:

    age_missing_before = df["age"].isnull().sum()

    median_age = df["age"].median()

    df["age"] = df["age"].fillna(median_age)


# ==========================================
# 12. HANDLE MISSING FARE
# ==========================================

fare_missing_before = 0

if "fare" in df.columns:

    fare_missing_before = df["fare"].isnull().sum()

    median_fare = df["fare"].median()

    df["fare"] = df["fare"].fillna(median_fare)


# ==========================================
# 13. HANDLE MISSING EMBARKED
# ==========================================

embarked_missing_before = 0

if "embarked" in df.columns:

    embarked_missing_before = df["embarked"].isnull().sum()

    if not df["embarked"].mode().empty:

        mode_value = df["embarked"].mode()[0]

        df["embarked"] = df["embarked"].fillna(
            mode_value
        )


# ==========================================
# 14. REMOVE CABIN COLUMN
# ==========================================

cabin_missing = 0

if "cabin" in df.columns:

    cabin_missing = df["cabin"].isnull().sum()

    df = df.drop(columns=["cabin"])


# ==========================================
# 15. FINAL CHECK
# ==========================================

print("\n========== CLEANED DATA ==========")

print("\nFirst 5 rows:")
print(df.head())

print("\nFinal shape:")
print(df.shape)

print("\nFinal data types:")
print(df.dtypes)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nRemaining duplicates:")
print(df.duplicated().sum())


# ==========================================
# 16. SAVE CLEANED DATASET
# ==========================================

df.to_csv(
    "data/titanic_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


# ==========================================
# 17. CREATE REPORT
# ==========================================

with open(
    "reports/data_quality_report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "SWYNEX DATA CLEANING & PREPARATION REPORT\n"
    )

    report.write("=" * 50 + "\n\n")

    report.write("Dataset: Titanic Dataset\n\n")

    report.write("CLEANING OPERATIONS\n")
    report.write("-" * 30 + "\n")

    report.write(
        f"Duplicate records removed: {duplicates}\n"
    )

    report.write(
        f"Invalid age values corrected: {invalid_age}\n"
    )

    report.write(
        f"Invalid fare values corrected: {invalid_fare}\n"
    )

    report.write(
        f"Missing age values before cleaning: "
        f"{age_missing_before}\n"
    )

    report.write(
        f"Missing fare values before cleaning: "
        f"{fare_missing_before}\n"
    )

    report.write(
        f"Missing embarked values before cleaning: "
        f"{embarked_missing_before}\n"
    )

    report.write(
        f"Missing cabin values: {cabin_missing}\n"
    )

    report.write(
        "Cabin column was removed because it "
        "contained many missing values.\n"
    )

    report.write("\nFINAL DATASET\n")
    report.write("-" * 30 + "\n")

    report.write(
        f"Rows: {len(df)}\n"
    )

    report.write(
        f"Columns: {len(df.columns)}\n"
    )

    report.write(
        f"Remaining duplicates: "
        f"{df.duplicated().sum()}\n"
    )

    report.write("\nRemaining missing values:\n")

    report.write(
        str(df.isnull().sum())
    )


print("\n====================================")
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("====================================")