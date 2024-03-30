import csv

def read_metrics_from_csv(csv_file):
    # Read existing metrics from the CSV file
    rows = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows