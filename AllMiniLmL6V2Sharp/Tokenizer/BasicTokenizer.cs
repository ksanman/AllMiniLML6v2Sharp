using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace AllMiniLmL6V2Sharp.Tokenizer
{
    public class BasicTokenizer : BaseTokenizer
    {
        private readonly bool IsLowerCase = true;
        public BasicTokenizer(bool isLowerCase)
        {
            IsLowerCase = isLowerCase;
        }

        public override IEnumerable<string> Tokenize(string text)
        {
            string cleanedText = CleanText(text);
            string cleanedChineseText = TokenizeChineseChars(cleanedText);
            IEnumerable<string> cleanedWhitespace = WhitespaceTokenize(cleanedChineseText);
            List<string> splitTokens = new List<string>();
            foreach (string word in cleanedWhitespace)
            {
                string token = word;
                if (IsLowerCase)
                {
                    string lower = token.ToLower();
                    token = RunStripAccents(lower);
                }

                IEnumerable<string> splitPunctuation = RunSplitOnPunc(token);
                splitTokens.AddRange(splitPunctuation);
            }

            IEnumerable<string> outputTokens = WhitespaceTokenize(string.Join(" ", splitTokens));
            return outputTokens;

        }

        private string CleanText(string text)
        {
            StringBuilder output = new StringBuilder();
            foreach (char c in text)
            {
                int charValue = CharUnicodeInfo.GetDigitValue(c);
                if (charValue == 0 || charValue == 0xfffd || IsControl(c))
                {
                    continue;
                }

                if (char.IsWhiteSpace(c))
                {
                    output.Append(' ');
                }
                else
                {
                    output.Append(c);
                }
            }

            return output.ToString();
        }

        private string TokenizeChineseChars(string text)
        {
            StringBuilder output = new StringBuilder();
            foreach (char c in text)
            {
                int charValue = CharUnicodeInfo.GetDigitValue(c);
                if (IsChineseChar(charValue))
                {
                    output.Append(' ');
                    output.Append(c);
                    output.Append(' ');
                }
                else
                {
                    output.Append(c);
                }
            }

            return output.ToString();
        }

        private bool IsChineseChar(int charValue)
        {
            return ((charValue >= 0x4E00 && charValue <= 0x9FFF) ||
                        (charValue >= 0x3400 && charValue <= 0x4DBF) ||
                        (charValue >= 0x20000 && charValue <= 0x2A6DF) ||
                        (charValue >= 0x2A700 && charValue <= 0x2B73F) ||
                        (charValue >= 0x2B740 && charValue <= 0x2B81F) ||
                        (charValue >= 0x2B820 && charValue <= 0x2CEAF) ||
                        (charValue >= 0xF900 && charValue <= 0xFAFF) ||
                        (charValue >= 0x2F800 && charValue <= 0x2FA1F));
        }

        private string RunStripAccents(string text)
        {
            string normalized = text.Normalize();
            StringBuilder output = new StringBuilder();
            foreach(char c in normalized)
            {
                UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(c);
                if(cat == UnicodeCategory.NonSpacingMark)
                {
                    continue;
                }

                output.Append(c);
            }

            return output.ToString();
        }

        private IEnumerable<string> RunSplitOnPunc(string text)
        {
            int i = 0;
            bool isStartOfNewWord = true;
            List<StringBuilder> output = new List<StringBuilder>();
            while(i < text.Length)
            {
                char c = text[i];
                if(IsPunctuation(c))
                {
                    output.Add(new StringBuilder().Append(c));
                    isStartOfNewWord = true;
                }
                else
                {
                    if (isStartOfNewWord)
                    {
                        output.Add(new StringBuilder());
                    }
                    isStartOfNewWord = false;
                    output.Last().Append(c);
                }
                i++;
            }

            return output.Select(o => o.ToString());
        }

        private bool IsControl(char c)
        {
            // These are technically control characters but we count them as whitespace
            // characters.
            if (c == '\t' || c == '\n' || c == '\r')
            {
                return false;
            }

            UnicodeCategory category = CharUnicodeInfo.GetUnicodeCategory(c);
            if (category == UnicodeCategory.Control || category == UnicodeCategory.Format)
            {
                return true;
            }

            return false;
        }

        private bool IsPunctuation(char c)
        {
            // We treat all non-letter/number ASCII as punctuation.
            // Characters such as "^", "$", and "`" are not in the Unicode
            // Punctuation class but we treat them as punctuation anyways, for
            // consistency.
            int charValue = CharUnicodeInfo.GetDigitValue(c);
            if ((charValue >= 33 && charValue <= 47) || (charValue >= 58 && charValue <= 64) ||
                    (charValue >= 91 && charValue <= 96) || (charValue >= 123 && charValue <= 126)) {
                return true;
            }

            return char.IsPunctuation(c);   
        }
    }
}
