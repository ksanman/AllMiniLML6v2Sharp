using AllMiniLmL6V2Sharp.Tokenizer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AllMiniLmL6V2Sharp.Tests
{
    public class TokenizerTests
    {
        private const string vocabPath = "./all-MiniLm-L6-v2/vocab.txt";
        [Theory]
        [InlineData("This is an example sentence")]
        [InlineData("Hello World!")]
        [InlineData("This is an example sentence.")]
        [InlineData("This is an example sentance")]
        [InlineData("sentance")]
        public void FullTokenizerTest(string sentence)
        {
            FullTokenizer tokenizer = new FullTokenizer(vocabPath);
            IEnumerable<Token> tokenized = tokenizer.Tokenize(sentence);
            Assert.NotNull(tokenized);
            Assert.NotEmpty(tokenized);
        }
    }
}
