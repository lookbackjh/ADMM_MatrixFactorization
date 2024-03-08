import torch
class Trainer():

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    
    def train(self, data, test_data, epochs):
        for epoch in range(epochs):
            for (x,y) in data:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                # print test loss
                test_loss = 0
                with torch.no_grad():
                    for (x,y) in test_data:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        output = self.model(x)
                        test_loss += self.criterion(output, y)

                #normalize test loss
                test_loss /= len(test_data)
                print(f"Epoch {epoch} train loss: {loss.item()} test loss: {test_loss.item()}")

        print("Training finished")

    

