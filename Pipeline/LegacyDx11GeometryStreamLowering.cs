using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Ruri.ShaderTools.Spirv;

namespace Ruri.ShaderTools.Pipeline;

/// <summary>
/// Stream-zero geometry instructions have the same emission semantics as the
/// corresponding non-stream instructions. The HLSL backend implements the latter
/// with TriangleStream.Append/RestartStrip, but emits unresolved GLSL calls for
/// the former. Normalize only a single geometry entry with proven integer zero
/// streams; never collapse multiple streams or replace source text.
/// Input is the valid IR produced by the existing frontend, not arbitrary
/// unvalidated SPIR-V; these checks do not replace a complete module validator.
/// </summary>
internal static class LegacyDx11GeometryStreamLowering
{
    private const uint GeometryExecutionModel = 3;
    private const uint StreamDecoration = 29;

    public static byte[] Apply(byte[] bytes)
    {
        ReadOnlySpan<uint> words = MemoryMarshal.Cast<byte, uint>(bytes.AsSpan());
        if (words.Length < SpirvModule.HeaderWordCount || bytes.Length % sizeof(uint) != 0
            || words[0] != SpirvModule.MagicNumber)
            throw new ArgumentException("Malformed SPIR-V header or byte length.");
        int entries = 0;
        bool geometry = false;
        var reader = new SpirvModule.InstructionReader(words);
        while (reader.MoveNext())
        {
            if (reader.OpCode != SpvOpCode.OpEntryPoint) continue;
            entries++;
            geometry |= reader.Current.Length >= 4 && reader.Current[1] == GeometryExecutionModel;
        }
        if (!reader.AtEnd)
            throw new ArgumentException("Malformed SPIR-V instruction length.");
        if (!geometry)
            return bytes;
        if (entries != 1)
            throw new ArgumentException("Single-entry geometry module required for stream lowering.");
        var integerTypes = new HashSet<uint>();
        var constants = new Dictionary<uint, (uint Type, uint Value)>();
        reader = new SpirvModule.InstructionReader(words);
        while (reader.MoveNext())
        {
            var instruction = reader.Current;
            if (reader.OpCode == SpvOpCode.OpTypeInt && instruction.Length == 4 && instruction[2] == 32 && instruction[3] <= 1)
                if (!integerTypes.Add(instruction[1]))
                    throw new ArgumentException("Duplicate SPIR-V integer type identity.");
            if (reader.OpCode == SpvOpCode.OpConstant && instruction.Length == 4)
                if (!constants.TryAdd(instruction[2], (instruction[1], instruction[3])))
                    throw new ArgumentException("Duplicate SPIR-V constant identity.");
            // Explicit output Stream decorations, including member decorations,
            // must agree with the stream-zero-only contract.
            if (reader.OpCode == SpvOpCode.OpDecorate && instruction.Length >= 3 && instruction[2] == StreamDecoration
                && (instruction.Length != 4 || instruction[3] != 0))
                throw new ArgumentException("Nonzero or malformed geometry Stream decoration.");
            if (reader.OpCode == SpvOpCode.OpMemberDecorate && instruction.Length >= 4 && instruction[3] == StreamDecoration
                && (instruction.Length != 5 || instruction[4] != 0))
                throw new ArgumentException("Nonzero or malformed geometry member Stream decoration.");
        }
        int changes = 0;
        reader = new SpirvModule.InstructionReader(words);
        while (reader.MoveNext())
        {
            if (!IsStreamInstruction(reader.OpCode)) continue;
            var instruction = reader.Current;
            if (instruction.Length != 2 || !constants.TryGetValue(instruction[1], out var value)
                || !integerTypes.Contains(value.Type) || value.Value != 0)
                throw new ArgumentException("Geometry stream must be a proven int32/uint32 literal zero.");
            changes++;
        }
        if (changes == 0) return bytes;
        // Validate every site before serializing anything. Keep instruction order,
        // constants, all output stores and all other instructions byte-for-byte.
        var result = new byte[bytes.Length - changes * sizeof(uint)];
        int cursor = SpirvModule.HeaderWordCount * sizeof(uint);
        bytes.AsSpan(0, cursor).CopyTo(result);
        reader = new SpirvModule.InstructionReader(words);
        while (reader.MoveNext())
        {
            var opCode = reader.OpCode;
            if (IsStreamInstruction(opCode))
            {
                ushort single = opCode == SpvOpCode.OpEmitStreamVertex ? SpvOpCode.OpEmitVertex : SpvOpCode.OpEndPrimitive;
                BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(cursor), SpvOpCode.MakeInstructionWord(single, 1));
                cursor += sizeof(uint);
            }
            else
            {
                var instructionBytes = MemoryMarshal.AsBytes(reader.Current);
                instructionBytes.CopyTo(result.AsSpan(cursor));
                cursor += instructionBytes.Length;
            }
        }
        return result;
    }

    private static bool IsStreamInstruction(ushort opCode)
        => opCode is SpvOpCode.OpEmitStreamVertex or SpvOpCode.OpEndStreamPrimitive;
}
