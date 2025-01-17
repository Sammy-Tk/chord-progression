import os
import time

# Path to the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Parent directory
parent_dir = os.path.dirname(script_dir)

# Directory "static"
static_dir = os.path.join(parent_dir, 'static')

# File age threshold
age_threshold_minutes = 50 # in minutes
age_threshold = age_threshold_minutes * 60  # in seconds

def clean_up_old_files(directory, age_threshold):
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > age_threshold:
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")

# Run the cleanup function
if __name__ == "__main__":
    clean_up_old_files(static_dir, age_threshold)
