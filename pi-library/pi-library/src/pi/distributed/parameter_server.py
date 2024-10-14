import torch.distributed as dist

class ParameterServer:
    def __init__(self, model):
        self.model = model

    def broadcast_parameters(self):
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def receive_gradients(self, worker_rank):
        for param in self.model.parameters():
            grad = torch.zeros_like(param.data)
            dist.recv(grad, src=worker_rank)
            param.grad = grad

    def update_model(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()