import numpy

def get_word_error_rate(r, h):

    """
    Given two list of strings how many word error rate(insert, delete or substitution).
    """
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    result = float(d[len(r)][len(h)]) / len(r) * 100
    return result


if __name__ == '__main__':
    r = ['This is the example :)', 'That is the example .']
    h = ['This is an example .', 'This is another example .']
    from utils.utils import ByteTextEncoder
    decoder = ByteTextEncoder(3)
    ids = decoder.encode(h[0])
    print(ids)
    s = decoder.decode(ids)
    print(s)
    wer = get_word_error_rate(r[0].split(), s.split())
    cer = get_word_error_rate(r[0], s)
    print(wer, cer)