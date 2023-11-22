namespace AllMiniLmL6V2Sharp.Tests
{
    public class AllMiniLmL6V2Tests
    {
        [Fact]
        public void ModelTest()
        {
            var model = new AllMiniLmL6V2();
            var sentence = "This is an example sentence";
            var embedding = model.Run(sentence);
            Assert.NotNull(embedding);
            Assert.NotEmpty(embedding);
        }
    }
}