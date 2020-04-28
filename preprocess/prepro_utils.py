import json
import string
import logging
import unicodedata
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

logger = logging.getLogger()


# ----------------------------------------------------------------------------------------
# clean and refactor source OpenKP dataset
# ----------------------------------------------------------------------------------------

def clean_phrase(Keyphrases):
    '''remove empty, duplicate and punctuation, do lower case'''
    def lower(text):
        return text.lower()
    def remove_punc(words):
        strings = ' '.join(words)
        return strings.strip('.').strip(',').strip('?').strip()
#         exclude = set(string.punctuation)
#         return ' '.join([w for w in words if w not in exclude])
    phrase_set = set()
    return_phrases = []
    for phrase in Keyphrases:
        if len(phrase) > 0:
            clean_phrase = lower(remove_punc(phrase))
            if len(clean_phrase) > 0 and clean_phrase not in phrase_set:
                return_phrases.append(clean_phrase.split())
                phrase_set.add(clean_phrase)
    return return_phrases


def refactor_text_vdom(text, VDOM):
    shuffler = DEL_ASCII()
    words = text.split()
    
    word2block, block_words, block_features = split_vdom(VDOM)
    
    doc_words = []
    new_word2block = []
    for wi, w in enumerate(words):
        if shuffler.do(w.strip()):
            doc_words.append(w.strip())
            bi = word2block[wi]
            if w in block_words[bi]:
                new_word2block.append(bi)
            else:
                new_word2block.append(-1)
                logger.info('Error ! the word not found in block')
                
    assert len(doc_words) == len(new_word2block)
    return doc_words, new_word2block, block_features


def split_vdom(VDOM):
    '''used in function `refactor_text_vdom` '''
    
    # convert string to json
    VDOMs = json.loads(VDOM)
    
    word2block = {}
    block_words = []
    block_features = []
    for bi, block in enumerate(VDOMs):
        block_words.append(block['text'].split())
        block_features.append(block['feature'])
        for wi in range(block['start_idx'], block['end_idx']):
            word2block[wi] = bi
            
    assert len(block_words) == len(block_features)
    return word2block, block_words, block_features


class DEL_ASCII(object):
    ''' used in function `refactor_text_vdom` for filter ASCII character b'\xef\xb8\x8f' '''
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
# find keyphrase position in document
# ----------------------------------------------------------------------------------------
def find_answer(document, answers):
    ''' 
    merge the same answers & keep present answers in document
    Inputs:
        document : a word list : ['sun', 'sunshine', ...] || lower cased
        answers : can have more than one answer : [['sun'], ['key','phrase'], ['sunshine']] || not duplicate
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






# ****************************************************************************************
# KP20k
# ****************************************************************************************
def find_stem_answer(word_list, ans_list):

    norm_doc_char, stem_doc_char = norm_doc_to_char(word_list)
    norm_stem_phrase_list = norm_phrase_to_char(ans_list)

    tot_ans_str = []
    tot_start_end_pos = []

    for norm_ans_char, stem_ans_char in norm_stem_phrase_list:    

        norm_stem_doc_char = " ".join([norm_doc_char, stem_doc_char])

        if norm_ans_char not in norm_stem_doc_char and stem_ans_char not in norm_stem_doc_char:
            continue
        else:
            norm_doc_words = norm_doc_char.split(" ")
            stem_doc_words = stem_doc_char.split(" ")

            norm_ans_words = norm_ans_char.split(" ")
            stem_ans_words = stem_ans_char.split(" ")

            assert len(norm_doc_words) ==len(stem_doc_words)
            assert len(norm_ans_words) == len(stem_ans_words)

            # find postions                
            tot_pos = []

            for i in range(0, len(stem_doc_words) - len(stem_ans_words) + 1):

                Flag = False

                if norm_ans_words == norm_doc_words[i:i+len(norm_ans_words)]:
                    Flag = True

                elif stem_ans_words == norm_doc_words[i:i+len(stem_ans_words)]:
                    Flag = True

                elif norm_ans_words == stem_doc_words[i:i+len(norm_ans_words)]:
                    Flag = True

                elif stem_ans_words == stem_doc_words[i:i+len(stem_ans_words)]:
                    Flag = True

                if Flag:
                    tot_pos.append([i, i+len(norm_ans_words)-1])
                    assert (i+len(stem_ans_words)-1) >= i

            if len(tot_pos) > 0 :
                tot_start_end_pos.append(tot_pos)
                tot_ans_str.append(norm_ans_char.split())

    assert len(tot_ans_str) == len(tot_start_end_pos)
    assert len(word_list) == len(norm_doc_char.split(" "))
    
    if len(tot_ans_str) == 0:
        return None
    return {'keyphrases':tot_ans_str, 'start_end_pos':tot_start_end_pos}

    
# used in function : find_answers
def norm_phrase_to_char(phrase_list):

    # del same keyphrase & null phrase
    norm_phrases = set()
    for phrase in phrase_list:
        p = " ".join([w.strip() for w in phrase if len(w.strip()) > 0])
        if len(p) < 1:continue
        norm_phrases.add(unicodedata.normalize('NFD', p))

    norm_stem_phrases = []
    for norm_chars in norm_phrases:
        stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
        norm_stem_phrases.append((norm_chars, stem_chars))

    return norm_stem_phrases


# used in function : find_answers
def norm_doc_to_char(word_list):

    norm_char = unicodedata.normalize('NFD', " ".join(word_list))
    stem_char = " ".join([stemmer.stem(w.strip()) for w in norm_char.split(" ")])

    return norm_char, stem_char
    