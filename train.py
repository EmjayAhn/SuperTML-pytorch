import copy
import time

import torch

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, epochs=20):
    start = time.time()
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        print("EPOCH {} / {}: ".format(epoch+1, epochs))
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            batch_loss = 0.0
            batch_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase =='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                batch_loss += loss.item() * inputs.size(0)
                batch_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = batch_loss / dataset_sizes[phase]
        epoch_acc = batch_corrects.double() / dataset_sizes[phase]
        
        print("{} Loss: {:.4f} Acc: {: .4f}".format(phase, epoch_loss, epoch_acc))
        
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            
    end = time.time()
    elapsed_time = end - start
    print("Training COMPLETED: {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("BEST VALIDATION ACCURACY: {:4f}".format(best_acc))
    
    model.load_state_dict(best_model_weights)
    return model