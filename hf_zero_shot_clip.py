import os
import sys
import torch
import torchvision
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

def custom_collate(batch, preprocessor, prompts):
    images, targets = zip(*batch)
    
    # Convert images to PIL format and apply CLIP preprocessing
    inputs = preprocessor(text=prompts, images=images, return_tensors="pt", padding=True)

    return inputs, torch.tensor(targets)
    
def main():
    parser = argparse.ArgumentParser(description='Process some data with a given model')
    parser.add_argument('--model-name', default='openai/clip-vit-base-patch32', help='Name of the model to use')
    parser.add_argument('--data-name', default='cifar10', help='Name of the data to process: cifar10 or cifar100')
    parser.add_argument('--device', default="cuda:3")
    
    args = parser.parse_args()

    model_name = args.model_name
    data_name = args.data_name
    device = args.device

    # load model
    model = CLIPModel.from_pretrained(model_name, device_map=device)
    preprocessor = CLIPProcessor.from_pretrained(model_name)

    # load dataset
    if data_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    elif data_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    else:
        raise ValueError("Data name is not supported")

    # define data_collator, x = batch data
    data_collator = lambda x: custom_collate(x, preprocessor, prompts)
    
    # define prompts
    prompts = [f"a photo of a {label}" for label in dataset.classes]
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=data_collator)

    # evaluating
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(dataloader)):
            # Get image features using CLIP
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # get image, text features
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            similarity = logits_per_image.softmax(dim=-1) # get cosine similarity between image and text features
            predicted_labels = similarity.argmax(dim=-1)
    
            # Update counts
            batch_correct = (predicted_labels == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            
            print(f"batch {idx}: correct {batch_correct}/{labels.shape[0]}")
            
    accuracy = correct / total
    
    print(f"Accuracy on {data_name} test set: {accuracy:.3f}")

if __name__ == "__main__":
    main()




