### Simple seq2seq aproach
In simple approach we tokenize toxic and non-toxic vocabulary, then learn how to extract meaningful data using `encode` network and restore sentence from meaningful data using `decode`. Since `encode` learns on toxic data, and `decode` learn on non-toxic data, `decode` does not know how to be toxic and, therefore conveys meaningful data without toxicity.

## GloVe
This approach is limited by simple tokenizer. Since each word corresponds to one token, teaching such model an entire dictionary is unpractical. Much better approach would be to make use of embedings, such as Global Vectors (GloVe). This way each word will be converted into a vector that represents it's meaning. This would allow model that only seen our dataset to understand and speak words that did not get into the traing set. 

## Pretrained networks
However this is not enough to guarantee preservation of meaning. Usage of pretrained models, such as T5 or Llama will give already reasonable neural network that we can get additional learning for. What's even better is that pretrained networks incorporate embeedings from the download.