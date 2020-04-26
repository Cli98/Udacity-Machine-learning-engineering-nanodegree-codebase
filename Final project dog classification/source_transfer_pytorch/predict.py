# import libraries
import os
import numpy as np
import torch
from six import BytesIO
import torchvision.models as models
import torch.nn as nn
# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-npy'

# Provided model load function
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_transfer = models.vgg19(pretrained=True)
    model_transfer.classifier[-1] = nn.Linear(4096, 133, bias=True)  
    if use_cuda:
        model_transfer = model_transfer.cuda()

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model_scratch.pth')
    with open(model_path, 'rb') as f:
        model_transfer.load_state_dict(torch.load(f,map_location='cpu'))

    # Prep for testing
    model_transfer.to(device).eval()

    print("Done loading model.")
    return model_transfer

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# Provided output data handling
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process input_data so that it is ready to be sent to our model.
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Put the model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data
    # The variable `out_label` should be a rounded value, either 1 or 0
    out = model(data)
    out_np = out.cpu().detach().numpy()
    # out_label = out_np.round()

    return out_np