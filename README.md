# AllMiniLmL6V2Sharp
C# implementation of Sentence Transformers All-MiniLM-L6-v2

Use as a .net standard 2.1 library.

Includes tokenizer and onnx model.

## Nuget
[AllMiniLML6v2Sharp](https://www.nuget.org/packages/AllMiniLmL6V2Sharp/)

The Nuget does not include the onnx model or the vocab.txt. These can be found on Hugging Face (See tested models below).
The Embedder looks for the default model.onnx and vocab.txt files in the ```.\model``` folder.
You may use a custom onnx model or custom vocab as well.

### How to use
- Single Sentence
```C#
var sentence = "This is an example sentence";
using var embedder = new AllMiniLmL6V2Embedder();
var embedding = embedder.GenerateEmbedding(sentence);
```
- Multiple Sentences
```C#
string[] sentences = ["This is an example sentence", "Here is another"];
using var embedder = new AllMiniLmL6V2Embedder();
var embeddings = model.GenerateEmbeddings(sentences);
```
- Custom All-MiniLM-L6-v2 onnx model
```C#
var sentence = "This is an example sentence";
using var embedder = new AllMiniLmL6V2Embedder(modelPath: "path/to/model.onnx");
var embedding = embedder.GenerateEmbedding(sentence);
```
- Custom vocab
```C#
var sentence = "This is an example sentence";
BertTokenizer tokenizer = new("path/to/vocab.txt");
using var embedder = new AllMiniLmL6V2Embedder(tokenizer: tokenizer);
var embedding = embedder.GenerateEmbedding(sentence);
```
- Custom Tokenizer
```C#
var sentence = "This is an example sentence";
ITokenizer tokenizer = new CustomTokenizer();
using var embedder = new AllMiniLmL6V2Embedder(tokenizer: tokenizer);
var embedding = embedder.GenerateEmbedding(sentence);
```

### Tested Models
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)