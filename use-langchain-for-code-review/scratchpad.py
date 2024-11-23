# Code in this file will be used as a placeholder to be submitted to LLMs for review

import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")

raw_text = "this is it"

# generate token IDs
encoded_text = torch.tensor(tokenizer.encode(raw_text))

print(f"Token IDs: {encoded_text}")
# Token IDs: tensor([1169,  530,  290,  691])

# we are going to have embedding dimension as 4
# for byte pair encoding vocabulary size in GPT2 is 50257
# let's create a token embedding layer
vocab_size = 50257
embedding_dimension = 4
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dimension)

# get token embeddings for Token IDs generated above
token_embeddings = token_embedding_layer(encoded_text)

print("Token Embeddings ---->")
print(token_embeddings)
# Token Embeddings ---->
# tensor([[ 0.0346, -2.4487,  0.2884, -0.8567],
#         [-0.9725, -0.9766,  0.1977,  2.2347],
#         [-0.4627, -0.0499,  0.8058, -0.5125],
#         [ 0.0799, -1.7682, -0.5987,  0.4784]], grad_fn=<EmbeddingBackward0>)

# Now let's compute weights about how each word in raw text relates to each other word in the same sentence
# dot product is an acceptable measure for similarity
# let's compute weights for dependency of words on each other by computing dot product
# row wise dot products
attention_weights = token_embeddings @ token_embeddings.T
print("attention_weights")
print(attention_weights)

# In the context of using PyTorch, the dim parameter in functions like torch.softmax specifies the dimension of the input
# tensor along which the function will be computed. By setting dim=-1, we are instructing the softmax function to apply
# the normalization along the last dimension of the attn_scores tensor. If attn_scores is a two-dimensional tensor
# (for example, with a shape of [rows, columns]), it will normalize across the columns so that the values in each row
# (summing over the column dimension) sum up to 1.
normalized_attention_weights = torch.softmax(attention_weights, dim=-1)
print("normalized_attention_weights")
print(normalized_attention_weights)
print(normalized_attention_weights.sum(dim=-1))


# attention weights multiplied to token embeddings (Notice in this calculation we are using token embeddings
# but ideally we should use token embeddings + positional embeddings)

all_context_vecs = normalized_attention_weights @ token_embeddings
print(all_context_vecs)