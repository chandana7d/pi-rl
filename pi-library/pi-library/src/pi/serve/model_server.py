class ModelServer:
    def __init__(self, model):
        self.model = model

    def serve(self, input_data):
        # Implement inference logic
        return self.model(input_data)
