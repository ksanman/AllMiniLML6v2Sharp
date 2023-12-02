using System.Collections.Generic;

namespace AllMiniLmL6V2Sharp.Tokenizer
{
    public interface ITokenizer
    {
        IEnumerable<Token> Tokenize(string text);
        IEnumerable<EncodedToken> Encode(int sequenceLength, string text);

    }
}
