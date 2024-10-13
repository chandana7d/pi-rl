class BaseAgent:
    def __init__(self):
        pass

    def act(self, state):
        raise NotImplementedError("This method should be overridden by subclasses")
