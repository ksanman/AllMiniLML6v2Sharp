namespace AllMiniLmL6V2Sharp.Tokenizer
{
    public class Token
    {
        public string Value { get; set; } = string.Empty;
        public long VocabularyIndex { get; set; }
        public long SegmentIndex { get; set; } 
    }
}
