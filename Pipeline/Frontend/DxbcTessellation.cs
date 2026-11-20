using System.Runtime.InteropServices;

namespace Ruri.ShaderTools.Pipeline.Frontend;

internal enum DxbcTessellationDomain { Isoline = 1, Triangle, Quad }

internal enum DxbcTessellationPartition { Integer = 1, PowerOfTwo, FractionalOdd, FractionalEven }

internal enum DxbcTessellationTopology { Point = 1, Line, TriangleClockwise, TriangleCounterClockwise }

internal readonly record struct DxbcTessellation(
    PipelineStage Stage,
    DxbcTessellationDomain Domain,
    int InputControlPoints,
    int OutputControlPoints,
    DxbcTessellationPartition Partition,
    DxbcTessellationTopology Topology,
    float? MaximumFactor)
{
    public static DxbcTessellation Read(ReadOnlySpan<byte> binary)
    {
        const uint extendedShader = 0x58454853; // SHEX
        const uint shader = 0x52444853; // SHDR
        const uint customData = 53;
        const uint declareInputPoints = 147;
        const uint declareOutputPoints = 148;
        const uint declareDomain = 149;
        const uint declarePartition = 150;
        const uint declareTopology = 151;
        const uint declareMaximumTessellationFactor = 152;
        ReadOnlySpan<byte> chunk = DxbcContainer.FindChunk(binary, [extendedShader, shader], out _);
        ReadOnlySpan<uint> words = MemoryMarshal.Cast<byte, uint>(chunk);
        if (chunk.Length % sizeof(uint) != 0 || words.Length < 2 || words[1] != words.Length)
            throw new InvalidDataException("Invalid DXBC tessellation instruction stream.");
        if ((words[0] & 0xffffu) != 0x50)
            throw new InvalidDataException("Expected an original Shader Model 5.0 DXBC tessellation program.");

        var stage = (words[0] >> 16) switch
        {
            3 => PipelineStage.TessControl,
            4 => PipelineStage.TessEvaluation,
            _ => throw new InvalidDataException("DXBC program is neither hull nor domain.")
        };

        int inputCount = 0;
        int outputCount = 0;
        int domain = 0;
        int partition = 0;
        int topology = 0;
        float? maximumFactor = null;
        for (int index = 2; index < words.Length;)
        {
            uint token = words[index];
            uint opcode = token & 0x7ff;
            uint length = opcode == customData && index + 1 < words.Length
                ? words[index + 1]
                : (token >> 24) & 0x7f;
            if (length < (opcode == customData ? 2u : 1u) || length > words.Length - index)
                throw new InvalidDataException("Invalid length of a DXBC instruction.");

            if (opcode >= declareInputPoints && opcode <= declareMaximumTessellationFactor)
            {
                if ((token & 0x80000000) != 0 || length != (opcode == declareMaximumTessellationFactor ? 2u : 1u))
                    throw new InvalidDataException("Invalid DXBC tessellation declaration.");
                int control = (int)(token >> 11);
                switch (opcode)
                {
                    case declareInputPoints: Set(ref inputCount, control & 0x3f, 32); break;
                    case declareOutputPoints: Set(ref outputCount, control & 0x3f, 32); break;
                    case declareDomain: Set(ref domain, control & 3, 3); break;
                    case declarePartition: Set(ref partition, control & 7, 4); break;
                    case declareTopology: Set(ref topology, control & 7, 4); break;
                    case declareMaximumTessellationFactor:
                        float value = BitConverter.UInt32BitsToSingle(words[index + 1]);
                        if (maximumFactor.HasValue || !(value >= 1 && value <= 64))
                            throw new InvalidDataException("Invalid DXBC maximum tessellation factor.");
                        maximumFactor = value;
                        break;
                }
            }
            index += (int)length;
        }

        bool hull = stage == PipelineStage.TessControl;
        if (inputCount == 0 || domain == 0 || (hull
            ? outputCount == 0 || partition == 0 || topology == 0
            : outputCount != 0 || partition != 0 || topology != 0 || maximumFactor.HasValue))
            throw new InvalidDataException("Incomplete or inconsistent DXBC tessellation declarations.");

        return new(stage, (DxbcTessellationDomain)domain, inputCount, outputCount,
            (DxbcTessellationPartition)partition, (DxbcTessellationTopology)topology, maximumFactor);
    }

    private static void Set(ref int destination, int value, int maximum)
    {
        if (destination != 0 || value < 1 || value > maximum)
            throw new InvalidDataException("Duplicate or out-of-range DXBC tessellation declaration.");
        destination = value;
    }
}
