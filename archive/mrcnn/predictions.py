
from . import prediction


class Predictions:
    """ wraps the predictions (masks=[100,1,N,N], etc.) with easy indexing """

    def __init__(self, p):
        """ stores vanilla predictions """
        self.data = p

    def __getitem__(self, index):
        """ easy indexing """
        return prediction.Prediction(
            mask=self.masks[index, 0].mul(255).byte(),
            box=self.boxes[index],
            label=self.labels[index],
            score=self.scores[index]
        )

    @property
    def boxes(self):
        return self.data['boxes']

    @property
    def labels(self):
        return self.data['labels']

    @property
    def masks(self):
        return self.data['masks']

    @property
    def scores(self):
        return self.data['scores']
