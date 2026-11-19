using System.Buffers.Binary;
using Ruri.ShaderTools.Pipeline.Backend;

namespace Ruri.ShaderTools.Unity.Smolv;

public static class UnityVulkanProgram
{
    public static IReadOnlyList<(PipelineStage Stage, byte[] Spirv)> Read(byte[] program, bool extendedTable)
    {
        // Unity's Vulkan blob starts with requirements, then offset/size pairs relative to the blob.
        // AssetStudio's SpirVShaderConverter documents this table and the extra 2019.3+ slot.
        const int entrySize = 2 * sizeof(int);
        Span<(int Offset, int Size)> regions = stackalloc (int, int)[extendedTable ? 6 : 5];
        var bytes = program.AsSpan();
        int tableEnd = sizeof(int);
        int dataStart = bytes.Length;
        int count = 0;
        for (int slot = 0; slot < regions.Length && tableEnd < dataStart; slot++)
        {
            if (dataStart - tableEnd < entrySize)
                throw new InvalidDataException("Truncated Unity Vulkan program table.");
            int offset = BinaryPrimitives.ReadInt32LittleEndian(bytes[tableEnd..]);
            int size = BinaryPrimitives.ReadInt32LittleEndian(bytes[(tableEnd + sizeof(int))..]);
            tableEnd += entrySize;
            if (size == 0) continue;
            if (size < 0 || offset < tableEnd || offset > bytes.Length || size > bytes.Length - offset)
                throw new InvalidDataException("Unity Vulkan module exceeds the program bounds.");
            regions[count++] = (offset, size);
            dataStart = Math.Min(dataStart, offset);
        }

        var modules = new List<(PipelineStage Stage, byte[] Spirv)>(count);
        foreach (var (offset, size) in regions[..count])
        {
            byte[] spirv = SmolvDecoder.Decode(program, offset, size)
                ?? throw new InvalidDataException("Unity Vulkan SMOL-V module is invalid or incomplete.");
            PipelineStage stage = EntryPointResolver.Resolve(spirv, preferredName: null).Stage;
            if (stage == PipelineStage.Unknown)
                throw new InvalidDataException("Unity Vulkan module has no supported SPIR-V entry point.");
            if (modules.Any(module => module.Stage == stage))
                throw new InvalidDataException("Unity Vulkan program has more than one module for a stage.");
            modules.Add((stage, spirv));
        }
        return modules;
    }
}
