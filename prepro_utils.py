import re
import sys
import logging
import unicodedata
import numpy as np
from tqdm import tqdm

logger = logging.getLogger()


class IdxTag_Converter(object):
    ''' idx2tag : a tag list like ['O','B','I','E','U']
        tag2idx : {'O': 0, 'B': 1, ..., 'U':4}
    '''
    def __init__(self, idx2tag):
        self.idx2tag = idx2tag
        tag2idx = {}
        for idx, tag in enumerate(idx2tag):
            tag2idx[tag] = idx
        self.tag2idx = tag2idx
        
    def convert_idx2tag(self, index_list):
        tag_list = [self.idx2tag[index] for index in index_list]
        return tag_list
        
    def convert_tag2idx(self, tag_list):
        index_list = [self.tag2idx[tag] for tag in tag_list]
        return index_list

# ----------------------------------------------------------------------------------------
# delet null keyphrase
# ----------------------------------------------------------------------------------------
class DEL_ASCII(object):
    ''' used in function `tokenize_data` for filter ASCII character b'\xef\xb8\x8f' '''
    def do(self, text):
        orig_tokens = self.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            token = self._run_strip_accents(token)                
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = self.whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
    
    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
    
    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]
    
    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
        
# ----------------------------------------------------------------------------------------
# find keyphrase position in document and merge the same keyphrases
# ----------------------------------------------------------------------------------------
def merge_same(phrases):
    '''
    merge the same keyphrases & 
    lower the phrases for find positions & output cased keyprhase removed '.'
    Input :
        phrases : a cased phrase token list .
    Outputs :
        lower_phrases : uncased phrase token list ,
        used for find keyphrase positions in document . 
        prepro_phrases : keep cased keyphrase (remove '.' in keyphrase's tail )
    '''
    phrases_set = set()
    lower_phrases = []
    prepro_phrases = []
    for phrase in phrases:
        phrase_string = ' '.join(phrase).strip('.').strip(' ') # delete '.' in the tail
        lower_phrase = phrase_string.lower()

        if lower_phrase in phrases_set:
            continue
        phrases_set.add(lower_phrase)
        lower_phrases.append(lower_phrase.split())
        prepro_phrases.append(phrase_string.split())
    return lower_phrases, prepro_phrases
    
    
def filter_overlap(positions):
    '''delete overlap keyphrase positions'''
    previous_e = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if s <= previous_e:
            continue
        filter_positions.append(positions[i])
        previous_e = e
    return filter_positions


def find_answer(document, answers):
    ''' 
    merge the same answers & keep present answers in document
    Inputs:
        document : a word list : ['sun', 'sunshine', ...], lower cased
        answers : can have more than one answer : 
        [['sun'], ['key','phrase'], ['sunshine']], not duplicate.
    Outputs:
        all_present_answers : prensent keyphrases
        positions_for_all : start_end_posisiton for prensent keyphrases
        a present answer postions list : every present's positions in documents, 
        each answer can be presented in several postions .
        [[[0,0],[20,21]], [[1,1]]]
    '''
    tot_doc_char = ' '.join(document)
    
    positions_for_all = []
    all_present_answers = []
    for answer in answers:
        ans_string = ' '.join(answer)
        
        if ans_string not in tot_doc_char:
            continue
        else: 
            positions_for_each = []
            # find all positions for each answer
            for i in range(0, len(document) - len(answer) + 1):
                Flag = False
                if answer == document[i:i+len(answer)]:
                    Flag = True
                if Flag:
                    assert (i+len(answer)-1) >= i
                    positions_for_each.append([i, i+len(answer)-1])
        if len(positions_for_each) > 0 :
            positions_for_all.append(positions_for_each)
            all_present_answers.append(answer)
            
    assert len(positions_for_all) == len(all_present_answers)
    
    if len(all_present_answers) == 0:
        return None
    return {'keyphrases':all_present_answers, 'start_end_pos':positions_for_all}


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# verify preprocess
def verify_ex(ex):
    ''' If you want to check whether the preprocess is true, 
    use these functions :`check_tokenize` & `check_tag`
    '''
    if check_tokenize(ex) and check_tag(ex):
        return True
    return False

def check_tokenize(ex):
    if len(ex['tokens']) != len(ex['valid_mask']):
        logger.info('ERROR : lengths not equal between tokens & valid_mask !')
        return False
    if sum(ex['valid_mask']) != len(ex['orig_tokens']):
        logger.info('ERROR : lengths not equal between sum_of valid_mask & orig_tokens !')
        return False
    return True

def check_tag(ex):
    if sum(ex['valid_mask']) != len(ex['label']):
        logger.info('ERROR : lengths not equal between valid_mask & label!')
        return False
    chunk_postions, _ = get_chunk_positions(ex['label'])
    flatten_postions = chunk_postions.tolist()
    sorted_positions = sorted(flatten_postions, key=lambda x: x[0])
    if sorted_positions != ex['orig_start_end_pos']:
        logger.info('ERROR : sorted_positions not equal orig_start_end_pos!')
        return False
    return True
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# convert tag list to start and end positions for (verify & evaluate)

def get_chunk_positions(tag_list, score_list=None):
    '''
    Parameters : 
        tag_list : ['O', 'I', ...]
        score_list : [0.68, 0.56, ...]
        each score is the prob of the tag prediction from model output
    Output : 
        chunk_postions : each chunk's start and end position list [[Ps1,Pe1], ...]
        chunk_scores : avearge the chunks'tag scores for each chunck
    '''
    if score_list is None:
        score_list = [1.0] * len(tag_list)
    assert len(tag_list) == len(score_list)

    abs_p1 = 0
    chunk_postions = []
    chunk_scores = []
    while abs_p1 < len(tag_list):
        now_tag = tag_list[abs_p1]
        
        if now_tag not in ['U', 'B']:
            abs_p1 += 1
            continue

        elif now_tag == 'U':
            chunk_postions.append([abs_p1,abs_p1])
            chunk_scores.append(score_list[abs_p1])
            abs_p1 += 1
            continue

        elif now_tag == 'B':
            flag = False
            ps = pe = abs_p1
            main_list = tag_list[abs_p1:]
            # last tag == 'B'
            if len(main_list) == 1:
                p1_offset = 1
            for p1_offset in range(1, len(main_list)):
                if main_list[p1_offset] not in ['I','E']:
                    break
                if main_list[p1_offset] == 'E':
                    pe = abs_p1 + p1_offset
                    flag = True
                    break
                if main_list[p1_offset] == 'I':
                    continue
            if flag and pe >= ps:
                chunk_postions.append([ps,pe])
                chunk_scores.append(np.mean(score_list[ps:pe+1]))
            abs_p1 += p1_offset
            
    return np.array(chunk_postions), np.array(chunk_scores)
