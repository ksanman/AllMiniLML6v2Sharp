using AllMiniLmL6V2Sharp.Tokenizer;

namespace AllMiniLmL6V2Sharp.Tests
{
    public class TokenizerTests
    {
        private const string vocabPath = "./model/vocab.txt";
        [Theory]
        [InlineData("This is an example sentence")]
        [InlineData("Hello World!")]
        [InlineData("This is an example sentence.")]
        [InlineData("This is an example sentance")]
        [InlineData("sentance")]
        public void BertTokenizerTest(string sentence)
        {
            BertTokenizer tokenizer = new(vocabPath);
            IEnumerable<Token> tokenized = tokenizer.Tokenize(sentence);
            Assert.NotNull(tokenized);
            Assert.NotEmpty(tokenized);
        }
    }
}
