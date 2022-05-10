
def first_n(iterator, n, suspected_length=None):
    if suspected_length is not None and suspected_length < n:
        n = suspected_length
    i = 0
    while i < n:
        i += 1
        yield next(iterator)
