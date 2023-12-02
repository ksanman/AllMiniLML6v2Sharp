# AllMiniLmL6V2Sharp
C# implementation of Sentance Transformers All-MiniLM-L6-v2

Use as a .net standard 2.1 library.

Includes tokenizer and onnx model.

### How to use
- Single Sentance
```C#
var sentance = "This is an example sentance";
var embedder = new AllMiniLmL6V2Embedder();
var embedding = embedder.GenerateEmbedding(sentance);
```
- Multiple Sentances
```C#
string[] sentences = ["This is an example sentence", "Here is another"];
var embedder = new AllMiniLmL6V2Embedder();
var embeddings = model.GenerateEmbeddings(sentences);
```
- Custom All-MiniLM-L6-v2 onnx model
```C#
var sentance = "This is an example sentance";
var embedder = new AllMiniLmL6V2Embedder(modelPath: "path/to/model.onnx");
var embedding = embedder.GenerateEmbedding(sentance);
```
- Custom vocab
```C#
var sentance = "This is an example sentance";
BertTokenizer tokenizer = new("path/to/vocab.txt");
var embedder = new AllMiniLmL6V2Embedder(tokenizer: tokenizer);
var embedding = embedder.GenerateEmbedding(sentance);
```
- Custom Tokenizer
```C#
var sentance = "This is an example sentance";
ITokenizer tokenizer = new CustomTokenizer();
var embedder = new AllMiniLmL6V2Embedder(tokenizer: tokenizer);
var embedding = embedder.GenerateEmbedding(sentance);
```