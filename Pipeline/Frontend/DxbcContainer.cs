using System.Buffers.Binary;

namespace Ruri.ShaderTools.Pipeline.Frontend;

internal static class DxbcContainer
{
    public static bool IsComputeShaderModel50(ReadOnlySpan<byte> binary)
    {
        const uint extendedShader = 0x58454853; // SHEX
        const uint shader = 0x52444853; // SHDR
        const uint computeShader = 5;
        const uint majorMinorMask = 0xFF;
        const uint model50 = 0x50;
        var chunk = FindChunk(binary, [extendedShader, shader], out _);
        if (chunk.Length < sizeof(uint)) return false;
        uint version = ReadWord(chunk);
        return (version >> 16) == computeShader && (version & majorMinorMask) == model50;
    }

    public static ReadOnlySpan<byte> FindChunk(ReadOnlySpan<byte> binary, ReadOnlySpan<uint> tags, out uint tag)
    {
        const int sizeOffset = 24;
        const int countOffset = 28;
        const int tableOffset = 32;
        tag = 0;
        if (binary.Length < tableOffset || ReadWord(binary) != 0x43425844) return default;

        uint size = ReadWord(binary[sizeOffset..]);
        uint count = ReadWord(binary[countOffset..]);
        if (size > binary.Length || size < tableOffset || count > (size - tableOffset) / sizeof(uint))
            throw new InvalidDataException("Invalid DXBC container size or chunk table.");

        int tableEnd = tableOffset + (int)count * sizeof(uint);
        ReadOnlySpan<byte> found = default;
        int priority = tags.Length;
        for (int index = 0; index < (int)count; index++)
        {
            uint offset = ReadWord(binary[(tableOffset + index * sizeof(uint))..]);
            if (offset < tableEnd || offset > size - 2 * sizeof(uint))
                throw new InvalidDataException("Invalid DXBC chunk offset.");

            ReadOnlySpan<byte> chunk = binary[(int)offset..];
            uint chunkTag = ReadWord(chunk);
            uint length = ReadWord(chunk[sizeof(uint)..]);
            if (length > size - offset - 2 * sizeof(uint))
                throw new InvalidDataException("Invalid DXBC chunk length.");

            int candidate = tags.IndexOf(chunkTag);
            if (candidate == priority)
                throw new InvalidDataException("Duplicate DXBC chunk.");
            if (candidate < 0 || candidate > priority) continue;
            tag = chunkTag;
            priority = candidate;
            found = chunk.Slice(2 * sizeof(uint), (int)length);
        }
        return found;
    }

    public static uint ReadWord(ReadOnlySpan<byte> bytes) => BinaryPrimitives.ReadUInt32LittleEndian(bytes);
}
