namespace AllMiniLmL6V2Sharp.Tests
{
    public class AllMiniLmL6V2Tests
    {
        [Fact]
        public void ModelTest()
        {
            var model = new AllMiniLmL6V2();
            var sentence = "This is an example sentence";
            var embedding = model.GenerateEmbedding(sentence);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);
        }

        [Fact]
        public void ModelMultipleTest()
        {
            var model = new AllMiniLmL6V2();
            string[] sentences = ["This is an example sentence", "Here is another"];
            var embedding = model.GenerateEmbeddings(sentences);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);
        }
    }
}