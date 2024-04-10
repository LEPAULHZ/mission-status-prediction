import pandas as pd
import os

def record_metrics_to_csv(df, csv_file):
    # Append or create the CSV file with the DataFrame
    df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))

