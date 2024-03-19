# Example test use of Tokenizer

import time
from tokenizer import Tokenizer

tokenizer = Tokenizer()

with open('training_data.txt', 'r', encoding = 'utf-8') as f:
    train_data = f.read()

random_text = 'Hello friend, how are you doing today :)'

t0 = time.time()
tokenizer.train(train_data, vocab_size = 1024)
t1 = time.time()
print(f'Training took {t1 - t0:.2f}s')

encoded = tokenizer.encode(random_text)
decoded = tokenizer.decode(encoded)

print('Everything worked!' if random_text == decoded else 'Something went wrong :(')