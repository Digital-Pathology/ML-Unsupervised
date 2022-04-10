
def ensure_individual_image(t, batch_to_single_index=0):
    if len(t.shape) == 3:
        return t
    else:
        if t.shape[0] > 1:
            return t[batch_to_single_index]
        else:
            return t.squeeze(0)
