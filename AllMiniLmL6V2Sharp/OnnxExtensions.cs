using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;

public static class OnnxExtensions
{
    public static DenseTensor<T> Unsqueeze<T>(this DenseTensor<T> tensor, int axis)
    {
        var originalShape = tensor.Dimensions;
        var newData = tensor.ToArray();

        var newShape = originalShape.ToArray().ToList();
        var newAxis = axis + originalShape.Length + 1;
        newShape.Insert(newAxis, 1);

        var unsqueezedTensor = new DenseTensor<T>(newData, newShape.ToArray());

        return unsqueezedTensor;
    }

    public static DenseTensor<T> Expand<T>(this DenseTensor<T> input, int[] newShape)
    {
        var inputData = input.ToArray();
        var originalShape = input.Dimensions.ToArray();

        if (newShape.Length != originalShape.Length)
        {
            throw new ArgumentException("The length of newShape must be equal to the number of dimensions in the original tensor.");
        }

        var expandedData = new T[newShape.Aggregate((a, b) => a * b)];
        var index = new int[newShape.Length];
        var strides = new int[originalShape.Length];

        // Calculate the strides
        strides[originalShape.Length - 1] = 1;
        for (int i = originalShape.Length - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * originalShape[i + 1];
        }

        // Perform the expansion
        for (int i = 0; i < expandedData.Length; i++)
        {
            expandedData[i] = inputData[
                Enumerable.Range(0, originalShape.Length)
                          .Select(d => index[d] % originalShape[d])
                          .Select((d, j) => d * strides[j])
                          .Sum()
            ];

            // Update the index for the next iteration
            for (int j = 0; j < index.Length; j++)
            {
                index[j]++;
                if (index[j] == newShape[j])
                {
                    index[j] = 0;
                }
                else
                {
                    break;
                }
            }
        }

        return new DenseTensor<T>(expandedData, newShape.ToArray());
    }

    public static DenseTensor<float> ElementWiseMultiply(this DenseTensor<float> tensor1, DenseTensor<float> tensor2)
    {
        var resultData = new float[tensor1.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = tensor1.Buffer.Span[i] * tensor2.Buffer.Span[i];
        }

        return new DenseTensor<float>(resultData, tensor1.Dimensions);
    }

    public static DenseTensor<float> Sum(this DenseTensor<float> input, int? dim = null, bool keepdim = false)
    {
        if (dim == null || !dim.HasValue)
        {
            // If dim is null, sum across all dimensions
            return SumAll(input, keepdim);
        }
        else
        {
            // Sum along the specified dimension
            return SumAlongDimension(input, dim.Value, keepdim);
        }
    }

    private static DenseTensor<float> SumAlongDimension(DenseTensor<float> inputTensor, int dim, bool keepdim)
    {
        int[] dimensions = inputTensor.Dimensions.ToArray();
        int[] outputShape = keepdim ? dimensions : dimensions.Select((index, idx) => dim == idx ? -1 : index).Where(x => x != -1).ToArray();
        float[] result = new float[outputShape.Aggregate((a, b) => a * b)];

        SumAlongDimensionRecursive(inputTensor, dim, keepdim, dimensions, result, new int[dimensions.Length], 0);

        return new DenseTensor<float>(result, outputShape);
    }

    private static void SumAlongDimensionRecursive(DenseTensor<float> inputTensor, int dim, bool keepdim, int[] dimensions, float[] result, int[] indices, int depth)
    {
        if (depth == dimensions.Length)
        {
            float element = (float)inputTensor.GetValue(GetFlattenedIndex(indices, dimensions));
            int[] reducedIndices = indices.Select((index, idx) => dim == idx ? 0 : index).ToArray();
            int[] outputIndices = keepdim
                ? reducedIndices
                : reducedIndices.Where((index, idx) => dimensions[idx] != 1).ToArray();
            int resultIndex = GetFlattenedIndex(outputIndices, dimensions);
            result[resultIndex] = result[resultIndex] +  element;
        }
        else
        {
            for (int i = 0; i < dimensions[depth]; i++)
            {
                indices[depth] = i;
                SumAlongDimensionRecursive(inputTensor, dim, keepdim, dimensions, result, indices, depth + 1);
            }
        }
    }

    private static DenseTensor<float> SumAll(DenseTensor<float> input, bool keepdim)
    {
        var sumResult = new float[1];
        for (int i = 0; i < input.Buffer.Span.Length; i++)
        {
            sumResult[0] += input.Buffer.Span[i];
        }

        if (keepdim)
        {
            return new DenseTensor<float>(sumResult, new int[] { 1 });
        }
        else
        {
            return new DenseTensor<float>(sumResult, new int[] { });
        }
    }

    private static int GetFlattenedIndex(int[] indices, int[] dimensions)
    {
        int index = 0;
        int multiplier = 1;

        for (int i = indices.Length - 1; i >= 0; i--)
        {
            index += indices[i] * multiplier;
            multiplier *= dimensions[i];
        }

        return index;
    }

    public static DenseTensor<float> Clamp(this DenseTensor<float> input, float min = float.MinValue, float max = float.MaxValue)
    {
        var resultData = new float[input.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = Math.Min(Math.Max(input.Buffer.Span[i], min), max);
        }

        return new DenseTensor<float>(resultData, input.Dimensions);
    }

    public static DenseTensor<float> ElementWiseDivide(this DenseTensor<float> tensor1, DenseTensor<float> tensor2)
    {
        var resultData = new float[tensor1.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = tensor1.Buffer.Span[i] / tensor2.Buffer.Span[i];
        }

        return new DenseTensor<float>(resultData, tensor1.Dimensions);
    }

    public static DenseTensor<float> Normalize(this DenseTensor<float> input, int p, int dim)
    {
        var normalizedData = new float[input.Length];

        for (int i = 0; i < input.Length; i++)
        {
            normalizedData[i] = input.Buffer.Span[i] / Norm(input, p, dim, i);
        }

        return new DenseTensor<float>(normalizedData, input.Dimensions);
    }

    private static float Norm(DenseTensor<float> input, int p, int dim, int flatIndex)
    {
        var indices = GetIndex(flatIndex, input.Dimensions.ToArray(), input.Buffer.Span.Length);
        var sum = 0.0f;

        for (int i = 0; i < input.Dimensions[dim]; i++)
        {
            indices[dim] = i;
            sum += (float)Math.Pow(input[indices], p);
        }

        return (float)Math.Pow(sum, 1.0 / p);
    }

    private static int[] GetIndex(int index, int[] dimensions, int mul)
    {
        int[] res = new int[dimensions.Length];

        for (int i = dimensions.Length; i != 0; --i)
        {
            mul /= dimensions[i - 1];
            res[i - 1] = index / mul;
            if (res[i - 1] >= dimensions[i - 1]) throw new Exception("Invalid Index");
            index -= res[i - 1] * mul;
        }
        return res;
    }
}