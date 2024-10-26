import csv
import numpy as np
import cv2
import torch
from itertools import combinations

class GenericError(BaseException):pass

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

def custom_grayscale(frame, red_weight, green_weight, blue_weight):
    # Assuming frame is in RGB format with shape (height, width, 3)
    weights = np.array([red_weight, green_weight, blue_weight])
    grayscale_image = np.dot(frame[..., :3], weights)
    return grayscale_image

def process_frame_buffer(frame):
    x1, y1 = 90, 81  # Top-left corner
    x2, y2 = 174, 165  # Bottom-right corner
    cropped = frame[y1:y2, x1:x2]

    gray_image = custom_grayscale(cropped, 0.7, 0.3, 0.0)

    new_width, new_height = 21, 21
    
    resized = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # cv2.imwrite("output.bmp", resized)
    # Normalize pixel values to [0, 1]
    normalized_image = resized.astype(np.float32) / 255.0

    return normalized_image

def compute_actions(max_simultaneous_buttons=2):
    num_bits = 8

    # Single 1 options
    single_ones = [f"{1 << i:08b}" for i in range(num_bits)]

    # Two 1s combinations
    two_ones =  [f"{(1 << i) | (1 << j):08b}" for i, j in combinations(range(num_bits), max_simultaneous_buttons)]

    # Combine both lists
    return single_ones + two_ones

    
global_device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )