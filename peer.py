import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import grpc
import signal
import proto.modelservice_pb2 as pb2
import proto.modelservice_pb2_grpc as pb2_grpc
from time import time, sleep
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
import random
import fcntl
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"USING: {DEVICE}")

MIN_PEERS_REQUIRED = 1  # Minimum number of peers needed for aggregation
MAX_RETRY_ATTEMPTS = 3  # Maximum number of retry attempts for collecting models

running = True
channel = None
port = 0

def signal_handler(signum, frame):
    global running
    print("\nShutting down gracefully...")
    running = False
    if channel:
        channel.close()

class JSONDataset(Dataset):
    """
    A PyTorch Dataset for loading data from a JSON file.
    """
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        data = torch.tensor(sample["data"], dtype=torch.float32)  # Convert back to tensor
        label = torch.tensor(sample["label"], dtype=torch.long)  # Convert back to tensor
        return data, label



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)  # Binary classification output
        self.relu = nn.ReLU()
        # super(SimpleCNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(3136, 128)
        # self.fc2 = nn.Linear(128, 10)  # Binary classification output
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))  # Binary output
        x = self.fc2(x)  # Binary output
        
        return x.squeeze()
    



def train_model(dataloader, model, criterion, optimizer_fn=optim.Adam, epochs=5):
    
    model = model.to(DEVICE)
    optimizer = optimizer_fn(model.parameters())
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            oh_labels = nn.functional.one_hot(labels, 10).float()
            # print(inputs.shape, outputs.shape, labels.shape)
            
            loss = criterion(outputs, oh_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # print(predicted.shape, labels.shape)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")
        print(f"Train Accuracy: {correct / total:.4f}")
        # torch.save(model.state_dict(), f"model_epoch{epoch}.pth")

    return model

def evaluate(model, dataloader, criterion):
    """
    Evaluates the model on a test dataset.
    
    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.

    Returns:
        dict: A dictionary containing the average loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # labels = labels.view(-1, 1)  # Reshape labels for binary classification
            oh_labels = nn.functional.one_hot(labels, 10).float()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, oh_labels)
            total_loss += loss.item()
            
            # Compute predictions and accurac
            
            # Track loss and accuracy
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # print(predicted.shape, labels.shape)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return f"Validation Accuracy: {accuracy:.4f}"
    # return {"loss": avg_loss, "accuracy": accuracy}

def bad_file_completion(path, timeout=5):
    
    prev = -1
    start_time = time()
    
    while time() - start_time < timeout:
        curr = os.path.getsize(path)
        
        if curr == prev:
            return True
        
        prev = curr
        sleep(1)
        
    return False

def aggregate_models(agg_dir, base_model_class, timeout=10):
    """
    Aggregates model weights from multiple .pth files using Federated Averaging (FedAvg).
    
    Parameters:
        model_paths (list): List of file paths to the .pth files (one per client).
        base_model_class (torch.nn.Module): Class of the base model (used to initialize the aggregated model).
        DEVICE (str): Device to load and process the models ("cpu" or "cuda").
    
    Returns:
        aggregated_model (torch.nn.Module): The model with aggregated weights.
    """
    stime = time()
    # Load the state dictionaries from all models

    while time() - stime < timeout:
        if os.path.exists(f"./aggs/agg{port}.DONE"):
            try:
                agg_paths = [os.path.join(agg_dir, x) for x in os.listdir(agg_dir) if x.endswith('.pth')]
                for path in agg_paths:
                    print(path)
                if len(agg_paths) < MIN_PEERS_REQUIRED:
                    print(f"Not enough models to aggregate. Found {len(agg_paths)}, need {MIN_PEERS_REQUIRED}")
                    os.remove("./aggs/agg{port}.DONE")
                    return None
                
                print(f"Found {len(agg_paths)} models for aggregation")


                # valid_state_dicts = []
                # for path in agg_paths:
                #     try:
                #         state_dict = torch.load(path, map_location=DEVICE)
                #         valid_state_dicts.append(state_dict)
                #     except Exception as e:
                #         print(f"Error loading model from {path}: {e}")
                #         continue
                
                # if len(valid_state_dicts) < MIN_PEERS_REQUIRED:
                #     print(f"Not enough valid models after validation")
                #     os.remove(".DONE")
                #     return None
                state_dict = [torch.load(path, map_location=DEVICE, weights_only=False) for path in agg_paths]

                
                base_model = base_model_class().to(DEVICE)
                aggregated_state_dict = base_model.state_dict()
            
                num_models = len(state_dict)
                for state_dict in state_dict:
                    for key in aggregated_state_dict.keys():
                        aggregated_state_dict[key] += state_dict[key]
                
                # Average the weights
                for key in aggregated_state_dict.keys():
                    aggregated_state_dict[key] /= num_models
                
                # Load the aggregated weights into the base model
                base_model.load_state_dict(aggregated_state_dict)
                os.remove("./aggs/agg{port}.DONE")
                for path in agg_paths:
                    print(path)
                    if path != f"aggs/agg{port}/my_model{port}":
                        try:
                            os.remove(path)
                        except OSError as e:
                            print(f"Error removing {path}: {e}")
                return base_model
            
            except Exception as e:
                print(f"Error during aggregation: {e}")
                if os.path.exists("./aggs/agg{port}.DONE"):
                    os.remove("./aggs/agg{port}.DONE")
                return None
        
        print("TIMEOUT")
        print(f"TIME: {time() - stime}")
        print(f"TIME: {time() - stime < timeout}")
        return None

def collect_models(client, secret_key, num_models, timeout=10):
    """
    Calls the CollectModels gRPC function on the Go server.

    Args:
        client: The gRPC client object.
        secret_key (str): The secret key for authentication.
        num_models (int): The number of models to collect.
        timeout (int): Timeout in seconds for the gRPC call.

    Returns:
        int: The number of models actually collected.
    """
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            print(f"num_models: {num_models}")
            request = pb2.CollectModelsRequest(key=secret_key, num=num_models)
            response = client.CollectModels(request, timeout=timeout)
            if response.success:
                print(f"Successfully collected models: {response.success}.")
                return response.success
            else:
                print(f"Failed to collect models on attempt {attempt + 1}")
        except grpc.RpcError as e:
            print(f"gRPC error on attempt {attempt + 1}: {e.code()} - {e.details()}")

        if attempt < MAX_RETRY_ATTEMPTS - 1:
            sleep(2 ** attempt)
    return 0


def main():
    # lowkey could boot the go client at the start 
    # TODO this makes more sense with NLP tasks maybe

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    if len(sys.argv) > 2:
        dataset = JSONDataset(sys.argv[1])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        global port
        port = int(sys.argv[2])
        
    else:
        print("Usage: python prototype.py [data json] [port]")
        exit(1)

    criterion = nn.CrossEntropyLoss()
    model = SimpleCNN()
    
    # torch.save(model.state_dict(), f"my_model.pth")
    start = time() # agg every 30 seconds
    
    test_dataset = JSONDataset("test_data.json")
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    agg_dir = f"aggs/agg{port}"
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)


    server_address = f"localhost:{port}"

    global channel 
    channel =  grpc.insecure_channel(server_address)
    client = pb2_grpc.ModelServiceStub(channel)

    global running
    try:
        while running:
            
            end = time()
            
            if end - start > 15:
                # this would be an RPC
                # print("BLOCK and requesting other models to be pushed into the to_aggregate folder")
                # sleep(3)
                

                # Connect to the server
                try:

                        # Secret key and number of models to collect
                        secret_key = "secret"
                        num_models = 1

                        # Call the function
                        num_models_collected = collect_models(client, secret_key, num_models, timeout=10)

                        if num_models_collected > 0:
                            new_model = aggregate_models(agg_dir, SimpleCNN)
                            if new_model is not None:
                                model = new_model
                                print("Successfully aggregated models")
                            else:
                                print("Continuing with current model due to aggregation failure")
                        else:
                            print("Continuing with current model due to collection failure")
                        # print(f"Models collected?: {models_collected}")
                except Exception as e:
                    print(f"Error during collection: {e}")
                
                start = time()
                
            else:
                model = train_model(dataloader, model, criterion, epochs=1)
            
            print(evaluate(model, test_dataloader, criterion))

            model_path = f"./aggs/agg{port}/my_model{port}.pth"
            temp_path = f"./aggs/agg{port}/my_model{port}.tmp"
            try: 
                torch.save(model.state_dict(), temp_path)
                os.replace(temp_path, model_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)      
    finally:
        if channel:
            channel.close()  

if __name__ == "__main__":
    main()