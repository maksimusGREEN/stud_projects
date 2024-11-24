from torch import optim
from torch import nn 
import torch


def train_word2vec(model, train_data, valid_data, device, epochs=8, learning_rate=0.01, verbose=True):

    best_model_wts = None
    best_loss = float('inf')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_function = nn.CrossEntropyLoss()

    model = model.to(device) 

    for epoch in range(epochs):

        model.train()

        train_loss, val_loss = 0, 0

        for inputs_batch, outputs_batch in train_data:

            inputs_batch = inputs_batch.to(device)
            outputs_batch = outputs_batch.to(device)

            optimizer.zero_grad()

            y_train_logits = model(inputs_batch)
            loss = loss_function(y_train_logits, outputs_batch)
            
            loss.backward()
            optimizer.step()   

            train_loss += loss.item() 

        avg_train_loss = train_loss / len(train_data)

        with torch.inference_mode():
            for inputs_batch, outputs_batch in valid_data:

                inputs_batch = inputs_batch.to(device)
                outputs_batch = outputs_batch.to(device)
                
                # Evaluate the validation loss
                inputs_batch_logits = model(inputs_batch)
                loss = loss_function(inputs_batch_logits, outputs_batch)

                val_loss += loss.item()
            
        # Calculate average validation loss for the epoch
        avg_val_loss = val_loss / len(valid_data)

        if verbose: 
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: ", avg_train_loss, "|||", "Validation Loss: ", avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)

    return model
    