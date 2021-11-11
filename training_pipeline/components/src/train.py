
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import torch.onnx

### Add References
import argparse
from azureml.core import Run,Workspace
from azureml.core.model import Model

### Add run context for AML
run = Run.get_context()

### Parse incoming parameters
parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point", default="")
parser.add_argument("--num-epochs", type=int, dest="num_epochs", help="Number of epochs", default="")
#parser.add_argument("--model-name", type=str, dest="model_name", help="Name of the registered Model", default="")
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--train_output', type=str, dest="train_output", help="Data output folder", default="")

args = parser.parse_args()
data_path = args.data_folder
num_epochs = args.num_epochs
#model_name = args.model_name
learning_rate = args.learning_rate
momentum = args.momentum
train_output = args.train_output

### Prepare the dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            
            print(phase,running_loss,dataset_sizes[phase])
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Log the los / acc to AMLS
            run.log("{} Loss".format(phase), np.float(epoch_loss))
            run.log("{} Acc".format(phase), np.float(epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    
    run.log("accuracy", np.float(best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

# Save the model
torch.save(model_ft, './outputs/model.pth')
torch.save(model_ft, os.path.join(train_output,'model.pth'))


# Save the labels
with open('./outputs/labels.txt', 'w') as f:
    f.writelines(["%s\n" % item  for item in class_names])

# Save the labels
with open(os.path.join(train_output,'labels.txt'), 'w') as f:
    f.writelines(["%s\n" % item  for item in class_names])

#run.upload_file(name='model/pytorch/labels.txt', path_or_stream="./outputs/labels.txt")
#run.upload_file(name='model/pytorch/model.pth', path_or_stream="./outputs/model.pth")

#tags = { "Conference":"<demo>" }
#model = run.register_model(model_name=model_name+"-PyTorch", model_path='model/pytorch/', tags=tags)

#print("== PyTorch Model Registered")
#print('Name:', model.name)
#print('Version:', model.version)

# Export the model
#print("== Export model to onnx ==")
#x = torch.randn(1, 3, 224, 224).to(device)
#torch.onnx.export(model_ft,x,"./outputs/model.onnx")

#run.upload_file(name='model/onnx/model.onnx', path_or_stream="./outputs/model.onnx")

#model = run.register_model(model_name=model_name+"-onnx", model_path='model/onnx/', tags=tags)

print("== Done == ")






