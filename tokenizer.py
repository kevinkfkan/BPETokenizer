# Trainable Tokenizer 
# Did not add special tokens

import regex as re

class Tokenizer:

    def __init__(self):
        # Copy GPT-4 rules for splitting text
        self.split_rules = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")      

    def _bpe_counts(self, ids, counts = None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):

        new_ids = []
        i=0
        while i < len(ids)-1:
            if ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i+=1
        
        if i == len(ids) - 1:
            new_ids.append(ids[-1])

        return new_ids

    def train(self, text, vocab_size):

        assert vocab_size >= 256
        num_merges = vocab_size - 256
        assert num_merges < len(list(text.encode('utf-8')))

        text_chunks = re.findall(self.split_rules, text)
        ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]

        vocab = {idx: bytes([idx]) for idx in range(256)}
        merges = {}

        for i in range(num_merges):

            counts = {}
            for chunk_id in ids:
                self._bpe_counts(chunk_id, counts)
            
            pair = max(counts, key=counts.get)

            idx = 256 + i
            ids = [self._merge(chunk_id, pair, idx) for chunk_id in ids]

            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            merges[pair] = idx
        
        self.vocab = vocab
        self.merges = merges

    def _encode_chunk(self, text):

        text_bytes = text.encode('utf-8') 
        ids = list(text_bytes)
        
        while len(ids) >= 2:
            counts = self._bpe_counts(ids)
            pair = min(counts, key = lambda p: self.merges.get(p, float('inf')))

            if pair not in self.merges:
                break
            
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
            
        return ids
    
    def encode(self, text):

        text_chunks = re.findall(self.split_rules, text)

        ids = []
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk)
            ids.extend(chunk_ids)
        
        return ids
    
    def decode(self, ids):

        text_bytes = b''.join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8')

        return text