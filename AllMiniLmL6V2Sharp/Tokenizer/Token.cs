namespace AllMiniLmL6V2Sharp.Tokenizer
{
    public class Token
    {
        public Token(string value, long segmentIndex, long vocabularyIndex)
        {
            Value = value;
            VocabularyIndex = vocabularyIndex;
            SegmentIndex = segmentIndex;
        }

        public string Value { get; set; } = string.Empty;
        public long VocabularyIndex { get; set; }
        public long SegmentIndex { get; set; } 
    }
}
