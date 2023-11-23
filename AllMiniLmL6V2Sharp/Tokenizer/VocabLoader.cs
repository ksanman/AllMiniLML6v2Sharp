using System.Collections.Generic;
using System.Collections.Specialized;
using System.IO;

namespace AllMiniLmL6V2Sharp.Tokenizer
{
    public class VocabLoader
    {
        public static IOrderedDictionary Load(string path)
        {
            IOrderedDictionary vocab = new OrderedDictionary();
            int index = 0;
            IEnumerable<string> lines = File.ReadLines(path);
            foreach (string line in lines)
            {
                if(string.IsNullOrEmpty(line)) break;
                string trimmedLine = line.Trim();
                vocab.Add(trimmedLine, index);
                index++;    
            }

            return vocab;
        }
    }
}
