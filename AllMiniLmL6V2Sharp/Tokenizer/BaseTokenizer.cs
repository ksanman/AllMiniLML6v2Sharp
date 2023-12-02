using System.Collections.Generic;

namespace AllMiniLmL6V2Sharp.Tokenizer
{
    internal abstract class BaseTokenizer
    {
        public abstract IEnumerable<string> Tokenize(string text);
        protected IEnumerable<string> WhitespaceTokenize(string text)
        {
            string strippedText = text.Trim();
            if (string.IsNullOrEmpty(strippedText))
            {
                return new List<string>();
            }

            IEnumerable<string> tokens = text.Split(' ');
            return tokens;
        }
    }
}
