import csv

def record_metrics_to_csv(metrics_dict, csv_file):
    # Create or append to the CSV file with metrics
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics_dict.keys())
        if file.tell() == 0:  # Check if the file is empty
            writer.writeheader()  # Write header if it's empty
        writer.writerow(metrics_dict)