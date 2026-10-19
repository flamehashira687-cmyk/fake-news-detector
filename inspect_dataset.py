from pathlib import Path
import pandas as pd

files = list(Path(".").glob("*.csv"))

if not files:
    print("CSV file not found in:", Path.cwd())
else:
    for file in files:
        print("\nFILE:", file.name)
        df = pd.read_csv(file)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        print("\nMissing values:")
        print(df.isnull().sum())

        for col in df.columns:
            if any(word in col.lower() for word in ["label", "class", "target"]):
                print("\nLabel counts:")
                print(df[col].value_counts(dropna=False))
