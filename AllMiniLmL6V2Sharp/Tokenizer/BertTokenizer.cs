using System.Collections.Generic;
using System.Linq;

namespace AllMiniLmL6V2Sharp.Tokenizer
{
    public class BertTokenizer : ITokenizer
    {
        private readonly IDictionary<string, int> _vocab;
        private readonly BasicTokenizer _basicTokenizer;
        private readonly WordpieceTokenizer _wordpieceTokenizer;
        private readonly IDictionary<int, string> _invVocab;
        public BertTokenizer(string vocabFile, bool isLowerCase = true, string unknownToken = Tokens.UNKNOWN_TOKEN, int maxInputCharsPerWord = 200) 
        {
            _vocab = VocabLoader.Load(vocabFile);
            _invVocab = new Dictionary<int, string>();
            foreach(KeyValuePair<string, int> kv in _vocab)
            {
                _invVocab.Add(kv.Value, kv.Key);
            }
            _basicTokenizer = new BasicTokenizer(isLowerCase);
            _wordpieceTokenizer = new WordpieceTokenizer(_vocab, unknownToken, maxInputCharsPerWord);
        }

        public IEnumerable<Token> Tokenize(string text)
        {
            List<Token> outputTokens = new List<Token>()
            {
                new Token(Tokens.CLS_TOKEN, 0, _vocab[Tokens.CLS_TOKEN])
            };

            int segmentIndex = 0;
            foreach (string token in _basicTokenizer.Tokenize(text))
            {
                foreach(string subToken in _wordpieceTokenizer.Tokenize(token))
                {
                    var outputToken = new Token(subToken, segmentIndex, _vocab[subToken]);
                    outputTokens.Add(outputToken);

                    if(token == Tokens.SEPARATOR_TOKEN)
                    {
                        segmentIndex++;
                    }
                }
            }

            outputTokens.Add(new Token(Tokens.SEPARATOR_TOKEN, segmentIndex++, _vocab[Tokens.SEPARATOR_TOKEN]));

            return outputTokens;
        }

        public IEnumerable<EncodedToken> Encode(int sequenceLength, string text)
        {
            IEnumerable<Token> tokens = Tokenize(text);

            IEnumerable<long> padding = Enumerable.Repeat(0L, sequenceLength - tokens.Count());
            return tokens
                .Select(token => new EncodedToken { InputIds = token.VocabularyIndex, TokenTypeIds = token.SegmentIndex, AttentionMask = 1L })
                .Concat(padding.Select(p => new EncodedToken { InputIds = p, TokenTypeIds = p, AttentionMask = p }));
        }
    }
}
