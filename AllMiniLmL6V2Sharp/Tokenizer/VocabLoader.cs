using System.Collections.Generic;
using System.IO;

namespace AllMiniLmL6V2Sharp.Tokenizer
{
    public class VocabLoader
    {
        public static IDictionary<string, int> Load(string path)
        {
            IDictionary<string, int> vocab = new Dictionary<string, int>();
            int index = 0;
            IEnumerable<string> lines = File.ReadLines(path);
            foreach (string line in lines)
            {
                if(string.IsNullOrEmpty(line)) break;
                string trimmedLine = line.Trim();
                vocab.Add(trimmedLine, index++);
            }

            return vocab;
        }
    }
}
