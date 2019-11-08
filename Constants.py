PAD = 0
UNK = 100
BOS = 101
EOS = 102
DIGIT = 1

PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
BOS_WORD = '[CLS]'
EOS_WORD = '[SEP]'
DIGIT_WORD = '[unused0]'

Idx2Tag = ['O','B','I','E','U']

# 'O' : non-keyphrase
# 'B' : begin word of the keyphrase
# 'I' : middle word of the keyphrase
# 'E' : end word of the keyphrase
# 'U' : single word keyphrase
