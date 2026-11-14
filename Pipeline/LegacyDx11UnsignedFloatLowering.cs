using Ruri.ShaderTools.Spirv;
using Ruri.ShaderTools.Spirv.Analysis;

namespace Ruri.ShaderTools.Pipeline;

internal static class LegacyDx11UnsignedFloatLowering
{
    private const ushort ConvertUToF = 112, UConvert = 113;

    public static byte[] Apply(byte[] bytes)
    {
        // SpirvModule.Parse stops at malformed tails; do not normalize such input.
        if (bytes.Length < 20 || bytes.Length % 4 != 0)
            throw new ArgumentException("Malformed SPIR-V byte length.");
        int cursor = 20;
        while (cursor < bytes.Length)
        {
            int size = (int)(System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(cursor)) >> 16) * 4;
            if (size < 4 || size > bytes.Length - cursor)
                throw new ArgumentException("Malformed SPIR-V instruction length.");
            cursor += size;
        }
        SpirvModule module = SpirvModule.Parse(bytes);
        var definitions = new Dictionary<uint, SpirvInstruction>();
        foreach (SpirvInstruction instruction in module.Instructions)
        {
            int? slot = SpvInstructionTraits.GetResultIdIndex(instruction);
            if (slot.HasValue)
            {
                if (slot.Value >= instruction.WordCount || !definitions.TryAdd(instruction[slot.Value], instruction))
                    throw new ArgumentException("Malformed SPIR-V result id.");
            }
        }
        bool changed = false;
        foreach (SpirvInstruction conversion in module.Instructions)
        {
            if (conversion.OpCode != ConvertUToF || conversion.WordCount != 4
                || !definitions.TryGetValue(conversion[3], out SpirvInstruction? widening)
                || widening.OpCode != UConvert || widening.WordCount != 4
                || !definitions.TryGetValue(widening[3], out SpirvInstruction? original))
                continue;
            int? typeSlot = SpvInstructionTraits.GetResultTypeIdIndex(original);
            if (!typeSlot.HasValue || typeSlot.Value >= original.WordCount)
                continue;
            int lanes = Shape(definitions, original[typeSlot.Value], 21, 32, 0);
            if (lanes == 0 || Shape(definitions, widening[1], 21, 64, 0) != lanes
                || Shape(definitions, conversion[1], 22, 32, 0) != lanes)
                continue;
            uint wideId = widening[2];
            // Unknown semantic annotations are not ours to discard or reinterpret.
            if (module.Instructions.Any(i => IsAnnotated(i, wideId)))
                continue;
            conversion[3] = widening[3];
            // Literal-aware counter: a constant's bits must not masquerade as an SSA use.
            // Require this to be the ONLY live consumer. Genuine 64-bit graphs stay intact.
            LiveUseCounter uses = LiveUseCounter.Build(module);
            if (uses.HasLiveConsumer(wideId))
            {
                conversion[3] = wideId;
                continue;
            }
            uses.Retire(widening);
            foreach (SpirvInstruction instruction in module.Instructions)
                if (instruction.OpCode == 5 && instruction.WordCount >= 2 && instruction[1] == wideId)
                    instruction.MakeNop();
            changed = true;
        }
        // Unused Int64 type/capability declarations are left intact. Never prune true64.
        return changed ? module.ToBytes() : bytes;
    }

    private static bool IsAnnotated(SpirvInstruction instruction, uint id)
    {
        if (instruction.WordCount < 2) return false;
        return instruction.OpCode switch
        {
            71 or 72 or 73 or 5632 or 5633 => instruction[1] == id,
            74 => ContainsRange(instruction, 1, 1, id),
            75 => instruction[1] == id || ContainsRange(instruction, 2, 2, id),
            332 => instruction[1] == id || ContainsRange(instruction, 3, 1, id),
            _ => false,
        };
    }

    private static bool ContainsRange(SpirvInstruction instruction, int start, int step, uint id)
    {
        for (int i = start; i < instruction.WordCount; i += step)
            if (instruction[i] == id) return true;
        return false;
    }

    private static int Shape(Dictionary<uint, SpirvInstruction> definitions, uint id, ushort opcode, uint width, uint sign)
    {
        if (!definitions.TryGetValue(id, out SpirvInstruction? type)) return 0;
        int lanes = 1;
        if (type.OpCode == 23 && type.WordCount == 4)
        {
            lanes = (int)type[3];
            if (lanes < 2 || lanes > 4 || !definitions.TryGetValue(type[2], out type)) return 0;
        }
        return type.OpCode == opcode && type.WordCount == (opcode == 21 ? 4 : 3)
            && type[2] == width && (opcode != 21 || type[3] == sign) ? lanes : 0;
    }
}
