class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add_experience(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Implement sampling logic
        pass
