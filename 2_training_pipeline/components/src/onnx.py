import torch
import os
import torch.onnx
import shutil

### Add References
import argparse

### Parse incoming parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_assets_path", type=str, dest="input_assets_path", help="Folder where the PyTorch model is in.", default="")
parser.add_argument("--output_assets_path", type=str, dest="output_assets_path", help="Folder where to save the converted model", default="")

args = parser.parse_args()
input_assets_path = args.input_assets_path
output_assets_path = args.output_assets_path

print('Input:',args.input_assets_path)
print('Output',args.output_assets_path)

# Export the model
device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = torch.load(os.path.join(input_assets_path,"model.pth"),map_location=lambda storage, loc: storage)

print("== Export model to onnx ==")
x = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(model_ft,x,os.path.join(output_assets_path,"model.onnx"))

# Copy the labels
shutil.copy2(os.path.join(input_assets_path,"labels.txt"),os.path.join(output_assets_path,"labels.txt"))

print("== Done == ")