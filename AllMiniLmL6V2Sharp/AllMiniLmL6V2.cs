using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AllMiniLmL6V2Sharp
{

    public class AllMiniLmL6V2
    {
        private readonly FullTokenizer _tokenizer;
        private readonly string _modelPath;
        public AllMiniLmL6V2(string modelPath = "./model/model.onnx", FullTokenizer? tokenizer = null)
        {
            _tokenizer = tokenizer ?? new FullTokenizer("./model/vocab.txt");
            _modelPath = modelPath;
        }

        public float[] GenerateEmbedding(string sentence)
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
            var pooled = SingleMeanPooling(output.First(), attMaskOrtValue);

            // Normalize Embeddings
            var normalized = pooled.Normalize(p: 2, dim: 1);

            var result = normalized.ToArray();
            return result;
        }

        public float[][] GenerateEmbeddings(IEnumerable<string> sentences)
        {
            // Tokenize Input
            IEnumerable<IEnumerable<Token>> allTokens = new List<IEnumerable<Token>>();
            IEnumerable<IEnumerable<EncodedToken>> allEncoded = new List<IEnumerable<EncodedToken>>();
            
            foreach (var sentence in sentences)
            {
                IEnumerable<Token> tokens = _tokenizer.Tokenize(sentence);

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
            dimensions[0] = 1;
            long shape = dimensions[0] * dimensions[1] * dimensions[2];

            for (long i = 0; i < output.Length; i += shape)
            {
                float[] buffer = new float[shape];
                Array.Copy(output, i, buffer, 0, shape);
                DenseTensor<float> tokenTensor = new DenseTensor<float>(buffer, dimensions);
                DenseTensor<float> maskTensor = AttentionMaskToTensor(attentionMask);
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
