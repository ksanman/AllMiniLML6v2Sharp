using AllMiniLmL6V2Sharp.Tokenizer;

namespace AllMiniLmL6V2Sharp.Tests
{
    public class VocabTests
    {
        private const string vocabPath = "./all-MiniLm-L6-v2/vocab.txt";

        [Fact]
        public void LoadVocabTest()
        {
            var vocab = VocabLoader.Load(vocabPath);
            Assert.NotNull(vocab);
            Assert.NotEmpty(vocab);
        }
    }
}
