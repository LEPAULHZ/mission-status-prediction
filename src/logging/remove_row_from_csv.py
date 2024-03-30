import csv
import os

def remove_row_from_csv(csv_file, line_number):
    # Remove a specific row from the CSV file
    rows_to_keep = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for current_line_number, row in enumerate(reader, 1):
            if current_line_number != line_number:
                rows_to_keep.append(row)

    # Write the remaining rows to a new CSV file
    with open('temp_file.csv', 'w', newline='') as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=rows_to_keep[0].keys())
        writer.writeheader()
        writer.writerows(rows_to_keep)

    # Replace the original file with the temporary file
    os.replace('temp_file.csv', csv_file)
