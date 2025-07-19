import os
from PIL import Image

def convert_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    Converts bounding box coordinates to YOLO format for a single class ("weapon").

    Args:
    - xmin, ymin, xmax, ymax (int): Coordinates of the bounding box.
    - img_width, img_height (int): Dimensions of the image.

    Returns:
    - YOLO format string: <class_index> <x_center> <y_center> <width> <height>
    """
    class_index = 0  # Class index is always 0 for "weapon"
    
    # Calculate the center of the bounding box
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    
    # Calculate the width and height of the bounding box
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    # Return the YOLO format as a string
    return f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_files_in_folder(input_folder, output_folder, image_folder):
    """
    Processes all annotation files in a folder and converts them to YOLO format using respective image dimensions.

    Args:
    - input_folder (str): Path to the folder containing input files.
    - output_folder (str): Path to the folder where YOLO format files will be saved.
    - image_folder (str): Path to the folder containing the respective images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):  # Assuming your annotation files are in .txt format
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            # Get corresponding image path
            image_filename = os.path.splitext(filename)[0] + '.jpg'  # Assuming images are .jpg
            image_path = os.path.join(image_folder, image_filename)

            if not os.path.exists(image_path):
                print(f"Image {image_filename} not found. Skipping file {filename}.")
                continue
            
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            with open(input_file_path, 'r') as file:
                lines = file.readlines()

            yolo_annotations = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, xmin, ymin, xmax, ymax = map(int, parts)
                    yolo_annotation = convert_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height)
                    yolo_annotations.append(yolo_annotation)
            
            # Write YOLO annotations to a new file
            with open(output_file_path, 'w') as output_file:
                for annotation in yolo_annotations:
                    output_file.write(annotation + '\n')

# Usage
input_folder = 'Labels'  # Replace with your input annotation folder path
output_folder = 'out'  # Replace with your output folder path
image_folder = 'Images'  # Replace with the folder containing images

# Process all files in the input folder
process_files_in_folder(input_folder, output_folder, image_folder)
