import torch

# Paths to your model files
model_path1 = 'D:\gpt2\TensorRT-LLM\examples\gpt\data\gpt2-finetuned-gita-2\pytorch_model.bin'
model_path2 = 'D:\gpt2\TensorRT-LLM\examples\gpt\data\gpt2-finetuned-gita-bible-quran-3\pytorch_model.bin'

# Load the state dictionaries
state_dict1 = torch.load(model_path1, map_location='cpu')
state_dict2 = torch.load(model_path2, map_location='cpu')


merged_state_dict = {}
c = 0
for key in state_dict1:
    if key in state_dict2:
        # Average the weights
        merged_state_dict[key] = (state_dict1[key] + state_dict2[key]) / 2
    else:
        # Or handle the scenario where keys don't match
        c = c + 1
        print(f"{key} is not matched number {c}")
        pass

merged_model_path = 'merged_model.bin'
torch.save(merged_state_dict, merged_model_path)

