
import re

def clean_text(text):
    # This regex will match any character that is NOT a letter, number, or space
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text


def process_file(input_file_path, output_file_path):
    """
    Read the content from input_file_path, clean it, and write the cleaned content to output_file_path.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            text = input_file.read()
        
        cleaned_text = clean_text(text)
        
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(cleaned_text)
        
        print(f"Successfully cleaned and saved to {output_file_path}")
    except Exception as e:
        print(f"Error: {e}")

out_file = r"D:\gpt2\TensorRT-LLM\examples\gpt\data\train_out.txt"

process_file(r"D:\gpt2\TensorRT-LLM\examples\gpt\data\train.txt", out_file)
f = open(out_file, encoding="UTF-8")

text = f.read()

token = len(sorted(set(text)))
print("num of tokens", token)
print(len(text))