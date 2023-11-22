using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;

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
            // Calculate L2 norm along the specified dimension
            var norms = torch.norm(pooled, 1, true, 2);

            // Normalize embeddings
            var normalized_embeddings = pooled.div(norms);

            var result = normalized_embeddings.data<float>().ToArray();
            return new float[384];
        }

        public void Run(IEnumerable<string> sentances)
        {
            throw new NotImplementedException();
        }

        private torch.Tensor MeanPooling(IDisposableReadOnlyCollection<OrtValue>  modelOutput, OrtValue attentionMask)
        {
            OrtValue tokenValue = modelOutput.First();
            var tokenTypeAndShape = tokenValue.GetTensorTypeAndShape();
            var tokenShape = new ReadOnlySpan<int>(tokenTypeAndShape.Shape.Select(s => (int)s).ToArray());
            var token_embeddings = tokenValue.GetTensorDataAsSpan<float>(); //First element of model_output contains all token embeddings
            var tokenTensor = torch.tensor(token_embeddings.ToArray(), torch.ScalarType.Float32).reshape(tokenTypeAndShape.Shape);

            var maskTypeAndShape = attentionMask.GetTensorTypeAndShape();
            var maskShape = new ReadOnlySpan<int>(maskTypeAndShape.Shape.Select(s => (int)s).ToArray());
            var mask = attentionMask.GetTensorDataAsSpan<long>();
            var maskData = new ReadOnlySpan<float>(mask.ToArray().Select(m => (float)m).ToArray());
            var maskTensor = torch.tensor(maskData.ToArray(), torch.ScalarType.Float32);

            var expanded = maskTensor.unsqueeze(-1).expand(tokenTypeAndShape.Shape);

            var multiplied = tokenTensor.mul(expanded);

            var sum = multiplied.sum(1);

            var sumMask = expanded.sum(1);

            var clampedMask = sumMask.clamp(min: 1e-9f);

            var result = sum.div(clampedMask);

            return result;
        }
    }
}
