import csv

def ram_dict_from_csv(file_path):
    ram_dict = {}
    
    # Open and read the CSV file
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)  # Using DictReader to map columns to keys
        
        # Iterate through each row in the CSV
        for row in reader:
            label = row['Label']
            address = int(row['Address'], 16)  # Convert hex string to integer
            ram_dict[label] = address
    
    return ram_dict

def state_list_from_csv(file_path):
    states = []
    
    # Open and read the CSV file
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        
        # Iterate through each row in the CSV
        for row in reader:
            states.append(row['Label'])
    
    return states