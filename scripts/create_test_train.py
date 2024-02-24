import torch # we use PyTorch: https://pytorch.org


def dataset():

    file_path = r"D:\gpt2\TensorRT-LLM\examples\gpt\data\train_out.txt"  # Update this to the path of your file

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # here are all the unique characters that occur in this text

    # create a mapping from characters to integers
    # stoi = { ch:i for i,ch in enumerate(chars) }
    # itos = { i:ch for i,ch in enumerate(chars) }
    # encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    # decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # print(encode("hii there"))
    # print(decode(encode("hii there")))

    # data = torch.tensor(encode(text), dtype=torch.long)
    # print(data.shape, data.dtype)


    # Let's now split up the data into train and validation sets
    n = int(0.9*len(text)) # first 90% will be train, rest val
    train_data = text[:n]
    val_data = text[n:]

    return train_data, val_data


# Function to save tensor data to a file
def save_data(file_path, data):
    # Convert tensor to list of integers
    # data_as_list = tensor_data.tolist()
    # Convert list of integers to a string where each integer is separated by a space
    # data_as_str = ' '.join(map(str, data_as_list))
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)

train, test = dataset()


train_path = "D:\gpt2\TensorRT-LLM\examples\gpt\data/GBQ_train_split.txt"
test_path = "D:\gpt2\TensorRT-LLM\examples\gpt\data/GBQ_test_split.txt"

save_data(train_path, train)
save_data(test_path, test)

# with open(train_path, 'w', encoding='utf-8') as file:
#     for item in train_data:
#         file.write(item)

# with open(test_path, 'w', encoding='utf-8') as file:
#     for item in val_data:
#         file.write(item)
