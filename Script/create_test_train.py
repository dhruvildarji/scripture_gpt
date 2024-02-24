import torch # we use PyTorch: https://pytorch.org
import argparse

def dataset(file_path):
    """
    Function to load and split the data into training and validation sets.

    Args:
        file_path (str): Path to the file containing the dataset.

    Returns:
        tuple: A tuple containing train_data and val_data.
    """
    # Open the file and read its contents
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the data into train and validation sets
    n = int(0.9 * len(text))  # First 90% will be train, rest val
    train_data = text[:n]
    val_data = text[n:]

    return train_data, val_data



# Function to save data to a file
def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)

if __name__ = "__main__":

    parser = argparse.ArgumentParser(description="Main script")
    parser.add_argument("--input_file_path", type=str, help="Path to the input file for processing", required=True)
    parser.add_argument("--output_train_file_path", type=str, help="Path to save the processed output", required=True)
    parser.add_argument("--output_test_file_path", type=str, help="Path to save the processed output", required=True)
    args = parser.parse_args()
    
    train, test = dataset(args.input_file_path)
    
    save_data(args.output_train_file_path, train)
    save_data(args.output_test_file_path, test)

