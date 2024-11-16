import csv
import json
import hashlib
import re
from typing import List, Dict

CSV_FILE_PATH = 'lab_3/33.csv'
RESULT_PATH = 'lab_3/result.json'
VARIANT_NUMBER = 33

PATTERN = {
    "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    "http_status_message": r'^\d{3} [A-Za-z ]+$',
    "snils": r'^\d{11}$',
    "passport": r'^\d{2} \d{2} \d{6}$',
    "ip_v4": r'^(?:\d{1,3}\.){3}\d{1,3}$',
    "longitude": r'^-?\d+\.\d+$',
    "hex_color": r'^#[0-9a-fA-F]{6}$',
    "isbn": r'(\d{3}-)?\d-(\d{5})-(\d{3})-\d',
    "locale_code": r'^[a-z]{2}(-[a-z]{2})?$',
    "time": r'^\d{2}:\d{2}:\d{2}\.\d{6}$'
}


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Calculates the MD5 hash of a list of integer values.
    
    :param row_numbers: A list of integer row numbers from the CSV file where validation errors were found.
    :return: MD5 hash for verification through GitHub Actions.
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    Serializes the results of the lab work into a JSON file.
    
    :param variant: The variant number of your assignment.
    :param checksum: The checksum calculated by calculate_checksum().
    """
    result = {
        "variant": variant,
        "checksum": checksum
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as jsonfile:
        json.dump(result, jsonfile, ensure_ascii=False, indent=4)


def is_valid(row: dict) -> bool:
    """
    Validates the data in a row using regular expressions.
    
    :param row: A dictionary representing a row of data.
    :return: True if the data is valid, otherwise False.
    """
    for key, pattern in PATTERN.items():
        if key in row and not re.match(pattern, row[key]):
            return False
    return True


def load_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Loads data from a CSV file.
    
    :param file_path: The path to the CSV file.
    :return: A list of dictionaries representing the rows of the CSV file.
    """
    with open(file_path, newline='', encoding='utf-16') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        return [row for row in reader]


def find_invalid_rows(rows: List[Dict[str, str]]) -> List[int]:
    """
    Finds invalid rows in the provided list of rows.
    
    :param rows: A list of dictionaries representing the rows of data.
    :return: A list of indices of invalid rows.
    """
    invalid_rows = []
    for index, row in enumerate(rows):
        if not is_valid(row):
            invalid_rows.append(index)
    return invalid_rows


def main():
    rows = load_csv(CSV_FILE_PATH)

    invalid_rows = find_invalid_rows(rows)

    print(f"Number of invalid rows: {len(invalid_rows)}")

    checksum = calculate_checksum(invalid_rows)

    serialize_result(VARIANT_NUMBER, checksum)


if __name__ == "__main__":
    main()
