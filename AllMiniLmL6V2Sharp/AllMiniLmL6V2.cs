using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AllMiniLmL6V2Sharp
{
    public struct BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        public long[] TypeIds { get; set; }
    }

    public class AllMiniLmL6V2
    {
        private readonly FullTokenizer _tokenizer;
        private readonly string _modelPath;
        public AllMiniLmL6V2(string modelPath = "./all-MiniLm-L6-v2/model.onnx", string vocabPath = "./all-MiniLm-L6-v2/vocab.txt")
        {
            _tokenizer = new FullTokenizer(vocabPath);
            _modelPath = modelPath;
        }

        public float[] Run(string sentence)
        {
            // Tokenize Input
            IEnumerable<Token> tokens = _tokenizer.Tokenize(sentence);
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
            var pooled = MeanPooling(output, attMaskOrtValue);

            // Normalize Embeddings
            var normalized = pooled.Normalize(p: 2, dim: 1);

            var result = normalized.ToArray();
            return result;
        }

        public void Run(IEnumerable<string> sentances)
        {
            throw new NotImplementedException();
        }

        private DenseTensor<float> MeanPooling(IDisposableReadOnlyCollection<OrtValue>  modelOutput, OrtValue attentionMask)
        {
            OrtValue tokenValue = modelOutput.First();
            DenseTensor<float> tokenTensor = OrtToTensor<float>(tokenValue);
            DenseTensor<long> maskIntTensor = OrtToTensor<long>(attentionMask);
            var maskFloatData = maskIntTensor.Select(x => (float)x).ToArray();
            DenseTensor<float> maskTensor = new DenseTensor<float>(maskFloatData, maskIntTensor.Dimensions);
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
