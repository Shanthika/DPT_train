import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        parameters = parameters["state_dict"]

        self.load_state_dict(parameters,strict=False)
