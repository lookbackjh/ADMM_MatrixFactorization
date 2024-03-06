import torch
class Trainer():

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    
    def train(self, data, epochs):
        for epoch in range(epochs):
            for (x,y) in data:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        print("Training finished")

    

