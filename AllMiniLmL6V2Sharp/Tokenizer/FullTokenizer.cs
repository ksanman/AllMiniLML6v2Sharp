using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;

namespace AllMiniLmL6V2Sharp.Tokenizer
{

    public class FullTokenizer
    {
        private readonly IOrderedDictionary _vocab;
        private readonly BasicTokenizer _basicTokenizer;
        private readonly WordpieceTokenizer _wordpieceTokenizer;
        private readonly IDictionary<int, string> _invVocab;
        public FullTokenizer(string vocabFile, bool isLowerCase = true, string unknownToken = "[UNK]", int maxInputCharsPerWord = 200) 
        {
            _vocab = VocabLoader.Load(vocabFile);
            _invVocab = new Dictionary<int, string>();
            foreach(DictionaryEntry kv in _vocab)
            {
                _invVocab.Add((int)kv.Value, (string)kv.Key);
            }
            _basicTokenizer = new BasicTokenizer(isLowerCase);
            _wordpieceTokenizer = new WordpieceTokenizer(_vocab, unknownToken, maxInputCharsPerWord);
        }

        public IEnumerable<Token> Tokenize(string text)
        {
            List<string> splitTokens = new List<string>();
            foreach(string token in _basicTokenizer.Tokenize(text))
            {
                foreach(string subToken in _wordpieceTokenizer.Tokenize(token))
                {
                    splitTokens.Add(subToken);
                }
            }

            splitTokens = splitTokens.Prepend("[CLS]").Append("[SEP]").ToList();

            // TODO - would be better to not do this in 2 loops.
            List<Token> outputTokens = new List<Token>();
            int segmentIndex = 0;
            foreach (var token in splitTokens)
            {
                var outputToken = new Token
                {
                    Value = token,
                    SegmentIndex = segmentIndex,
                    VocabularyIndex = (int)_vocab[token]
                };
                outputTokens.Add(outputToken);

                if (token == "[SEP]")
                {
                    segmentIndex++;
                }
            }

            return outputTokens;
        }

        public IEnumerable<EncodedToken> Encode(int sequenceLength, string text)
        {
            IEnumerable<Token> tokens = Tokenize(text);

            var padding = Enumerable.Repeat(0L, sequenceLength - tokens.Count());

            IEnumerable<long> tokenIndexes = tokens.Select(token => token.VocabularyIndex).Concat(padding);
            IEnumerable<long> segmentIndexes = tokens.Select(token => token.SegmentIndex).Concat(padding);
            IEnumerable<long> inputMask = tokens.Select(o => 1L).Concat(padding);

            IEnumerable<Tuple<long, long, long>> output = tokenIndexes.Zip(segmentIndexes, Tuple.Create)
                .Zip(inputMask, (t, z) => Tuple.Create(t.Item1, t.Item2, z));

            return output.Select(x => new EncodedToken { InputIds = x.Item1, TokenTypeIds = x.Item2, AttentionMask = x.Item3 });
            
        }

        public IEnumerable<int> ConvertTokensToIds(IEnumerable<string> items)
        {
            List<int> output = new List<int>();
            foreach(string item in items)
            {
                output.Add((int)_vocab[item]);
            }
            return output;
        }

        public IEnumerable<string> ConvertIdsToTokens(IEnumerable<int> items)
        {
            List<string> output = new List<string>();
            foreach(int id in items)
            {
                output.Add(ConvertIdToToken(id));
            }

            return output;
        }

        public string ConvertIdToToken(int id)
        {
            return _invVocab[id];
        }
    }
}
