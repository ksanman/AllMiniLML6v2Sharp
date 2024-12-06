using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AllMiniLmL6V2Sharp
{
    /// <summary>
    /// Generate Embeddings via All-MiniLM-L6-v2
    /// </summary>
    public class AllMiniLmL6V2Embedder : IEmbedder
    {
        private readonly ITokenizer _tokenizer;
        private readonly string _modelPath;
        private readonly bool _truncate;
        /// <summary>
        /// Initializes the AllMiniLmL6v2 Embedder
        /// </summary>
        /// <param name="modelPath">Path to the embedding onnx model.</param>
        /// <param name="tokenizer">Optional custom tokenizer function.</param>
        /// <param name="truncate">If true, automatically truncates tokens to 512 tokens.</param>
        public AllMiniLmL6V2Embedder(string modelPath = "./model/model.onnx", ITokenizer? tokenizer = null, bool truncate = false)
        {
            _tokenizer = tokenizer ?? new BertTokenizer("./model/vocab.txt");
            _modelPath = modelPath;
            _truncate = truncate;
        }

        /// <summary>
        /// Generates an embedding array for the given sentance.
        /// </summary>
        /// <param name="sentence">Text to embed.</param>
        /// <returns>Sentance embeddings</returns>
        public IEnumerable<float> GenerateEmbedding(string sentence)
        {
            // Tokenize Input
            IEnumerable<Token> tokens = _tokenizer.Tokenize(sentence);
            if(_truncate && tokens.Count() > 512)
            {
                tokens = tokens.Take(512);
            }

            IEnumerable<EncodedToken> encodedTokens = _tokenizer.Encode(tokens.Count(), sentence);

            // Compute Token Embeddings
            BertInput bertInput = new BertInput
            {
                InputIds = encodedTokens.Select(t => t.InputIds).ToArray(),
                TypeIds = encodedTokens.Select(t => t.TokenTypeIds).ToArray(),
                AttentionMask = encodedTokens.Select(t => t.AttentionMask).ToArray()
            };

            using RunOptions runOptions = new RunOptions();
            using InferenceSession session = new InferenceSession(_modelPath);

            // Create input tensors over the input data.
            using OrtValue inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                  new long[] { 1, bertInput.InputIds.Length });

            using OrtValue attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                  new long[] { 1, bertInput.AttentionMask.Length });

            using OrtValue typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
                  new long[] { 1, bertInput.TypeIds.Length });

            // Create input data for session. Request all outputs in this case.
            IReadOnlyDictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
                { "token_type_ids", typeIdsOrtValue }
            };

            using IDisposableReadOnlyCollection<OrtValue> output = session.Run(runOptions, inputs, session.OutputNames);

            // Perform Pooling
            var pooled = SingleMeanPooling(output.First(), attMaskOrtValue);

            // Normalize Embeddings
            var normalized = pooled.Normalize(p: 2, dim: 1);

            var result = normalized.ToArray();
            return result;
        }

        /// <summary>
        /// Generates an embedding array for the given sentances.
        /// </summary>
        /// <param name="sentence">Text to embed.</param>
        /// <returns>An enumerable of embeddings.</returns>
        public IEnumerable<IEnumerable<float>> GenerateEmbeddings(IEnumerable<string> sentences)
        {
            // Tokenize Input
            IEnumerable<IEnumerable<Token>> allTokens = new List<IEnumerable<Token>>();
            IEnumerable<IEnumerable<EncodedToken>> allEncoded = new List<IEnumerable<EncodedToken>>();
            
            foreach (var sentence in sentences)
            {
                IEnumerable<Token> tokens = _tokenizer.Tokenize(sentence);

                if(_truncate && tokens.Count() > 512)
                {
                    tokens = tokens.Take(512);
                }

                allTokens = allTokens.Append(tokens);
            }

            int maxSequence = allTokens.Max(t => t.Count());
            
            foreach(var sentence in sentences)
            {
                IEnumerable<EncodedToken> encodedTokens = _tokenizer.Encode(maxSequence, sentence);
                allEncoded = allEncoded.Append(encodedTokens);
            }

            // Compute Token Embeddings
            IEnumerable<BertInput> inputs = allEncoded.Select(e => new BertInput
            {
                InputIds = e.Select(t => t.InputIds).ToArray(),
                TypeIds = e.Select(t => t.TokenTypeIds).ToArray(),
                AttentionMask = e.Select(t => t.AttentionMask).ToArray()
            });

            using RunOptions runOptions = new RunOptions();
            using InferenceSession session = new InferenceSession(_modelPath);

            // Create input tensors over the input data.
            var size = inputs.Count();
            var inputIds = inputs.SelectMany(i => i.InputIds).ToArray();
            using OrtValue inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(inputIds,
                  new long[] { size, maxSequence });

            var attentionMask = inputs.SelectMany(i => i.AttentionMask).ToArray();
            using OrtValue attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(attentionMask,
                  new long[] { size, inputs.First().AttentionMask.Length });

            var typeIds = inputs.SelectMany(i => i.TypeIds).ToArray();
            using OrtValue typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(typeIds,
                  new long[] { size, maxSequence });

            // Create input data for session. Request all outputs in this case.
            IReadOnlyDictionary<string, OrtValue> ortInputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
                { "token_type_ids", typeIdsOrtValue }
            };

            using IDisposableReadOnlyCollection<OrtValue> output = session.Run(runOptions, ortInputs, session.OutputNames);

            // For now, perform this seperatly for each output value.
            return MultiplePostProcess(output.First(), attMaskOrtValue);
        }

        private float[][] MultiplePostProcess(OrtValue modelOutput, OrtValue attentionMask)
        {
            List<float[]> results = new List<float[]>();
            float[] output = modelOutput.GetTensorDataAsSpan<float>().ToArray();
            int[] dimensions = modelOutput.GetTensorTypeAndShape().Shape.Select(s => (int)s).ToArray();
            dimensions[0] = 1; // Since only processing 1 row at a time, set to 1. 
            long shape = dimensions[0] * dimensions[1] * dimensions[2];

            long[] mask = attentionMask.GetTensorDataAsSpan<long>().ToArray();
            int[] maskDimensions = attentionMask.GetTensorTypeAndShape().Shape.Select(s => (int)s).ToArray();
            maskDimensions[0] = 1; // Since only processing 1 row at a time, set to 1. 
            long maskShape = maskDimensions[0] * maskDimensions[1];
            int indicies = (int)Math.Floor((double)output.Length / (double)shape);

            for (long i = 0; i < indicies; i++)
            {
                long sourceIndex = shape * i;
                float[] buffer = new float[shape];
                Array.Copy(output, sourceIndex, buffer, 0, shape);
                DenseTensor<float> tokenTensor = new DenseTensor<float>(buffer, dimensions);

                long[] maskBuffer = new long[maskShape];
                long maskIndex = maskShape * i;
                Array.Copy(mask, maskIndex, maskBuffer, 0, maskShape);

                DenseTensor<float> maskTensor = new DenseTensor<float>(maskBuffer.Select(x => (float)x).ToArray(), maskDimensions);

                var pooled = MeanPooling(tokenTensor, maskTensor);
                // Normalize Embeddings
                var normalized = pooled.Normalize(p: 2, dim: 1);
                results.Add(normalized.ToArray());
            }

            return results.ToArray();
        }

        private DenseTensor<float> SingleMeanPooling(OrtValue modelOutput, OrtValue attentionMask)
        {
            DenseTensor<float> tokenTensor = OrtToTensor<float>(modelOutput);
            DenseTensor<float> maskTensor = AttentionMaskToTensor(attentionMask);
            return MeanPooling(tokenTensor, maskTensor);
        }

        private static DenseTensor<float> AttentionMaskToTensor(OrtValue attentionMask)
        {
            DenseTensor<long> maskIntTensor = OrtToTensor<long>(attentionMask);
            var maskFloatData = maskIntTensor.Select(x => (float)x).ToArray();
            DenseTensor<float> maskTensor = new DenseTensor<float>(maskFloatData, maskIntTensor.Dimensions);
            return maskTensor;
        }

        private DenseTensor<float> MeanPooling(DenseTensor<float> tokenTensor, DenseTensor<float> maskTensor)
        { 
            DenseTensor<float> maskedSum = ApplyMaskAndSum(tokenTensor, maskTensor);
            return maskedSum;
        }

        private DenseTensor<float> ApplyMaskAndSum(DenseTensor<float> tokenTensor, DenseTensor<float> maskTensor)
        {
            var expanded = maskTensor.Unsqueeze(-1).Expand(tokenTensor.Dimensions.ToArray());

            var multiplied = tokenTensor.ElementWiseMultiply(expanded);

            var sum = multiplied.Sum(1);

            var sumMask = expanded.Sum(1);

            var clampedMask = sumMask.Clamp(min: 1e-9f);

            var result = sum.ElementWiseDivide(clampedMask);

            return result;
        }

        private static DenseTensor<T> OrtToTensor<T>(OrtValue value) where T : unmanaged
        {
            var typeAndShape = value.GetTensorTypeAndShape();
            var tokenShape = new ReadOnlySpan<int>(typeAndShape.Shape.Select(s => (int)s).ToArray());
            var tokenEmbeddings = value.GetTensorDataAsSpan<T>();
            DenseTensor<T> tokenTensor = new DenseTensor<T>(tokenShape);
            tokenEmbeddings.CopyTo(tokenTensor.Buffer.Span);
            return tokenTensor;
        }
    }
}
