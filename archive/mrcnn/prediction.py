
class Prediction:
    def __init__(self, mask, box, label, score):
        self.mask = mask
        self.box = box
        self.label = label
        self.score = score
