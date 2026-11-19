using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Ruri.ShaderTools;

internal static class LegacyDx11ZeroEmissionGeometry
{
    // SM5 token values/layout: Microsoft DirectX-Headers, D3D12TokenizedProgramFormat.hpp.
    private const uint VertexShaderModel50 = (1u << 16) | (5u << 4);
    private const uint GeometryShaderModel50 = (2u << 16) | (5u << 4);
    private const uint Ret = 62, DeclareInputPrimitive = 93, DeclareMaxOutputVertices = 94;
    private const uint DeclareInput = 95, DeclareInputSystemValue = 97, DeclareGlobalFlags = 106;
    private const uint AllowRefactoring = 1u << 11, TrianglePrimitive = 3, TriangleVertices = 3;
    private const uint VertexInputOperand = 2u | (1u << 12) | (2u << 20);
    private const uint PrimitiveIdOperand = 11u << 12;
    private const uint Position = 1, PrimitiveId = 7;
    private const uint UInt32 = 1, SInt32 = 2, Float32 = 3;

    internal sealed record Element(string Semantic, uint Index, uint SystemValue, uint Register, uint Mask, string Type)
    {
        public bool IsPrimitiveId => SystemValue == PrimitiveId;
        public string SourceSemantic => SystemValue == 0 ? Semantic + Index.ToString(CultureInfo.InvariantCulture) : Semantic;
        // This proof admits only low-contiguous masks, so the first lane is zero.
        public string EmittedSemantic => SystemValue == 0 ? "TEXCOORD" + (Register * 4).ToString(CultureInfo.InvariantCulture) : Semantic;
        public bool Matches(Element other) => Index == other.Index && SystemValue == other.SystemValue && Register == other.Register
            && Mask == other.Mask && Type == other.Type && Semantic.Equals(other.Semantic, StringComparison.OrdinalIgnoreCase);
    }

    internal sealed record Proof(string Source, IReadOnlyList<Element> Inputs);

    public static Proof? Restore(ReadOnlySpan<byte> binary)
    {
        if (!TryGetChunks(binary, out var input, out var output, out var program)
            || output.Length != 8 || Read(output, 0) != 0 || Read(output, 4) != 8
            || !TryReadSignature(input, isInput: true, out var elements)
            || !TryReadReturnOnly(program, elements, out uint maxVertices))
            return null;

        var text = new StringBuilder("struct SPIRV_Cross_Input\n{\n");
        for (int i = 0; i < elements.Count; i++)
        {
            Element element = elements[i];
            if (element.IsPrimitiveId) continue;
            text.Append("    ").Append(element.Type).Append(" input_").Append(i.ToString(CultureInfo.InvariantCulture))
                .Append(" : ").Append(element.SourceSemantic).Append(";\n");
        }
        text.Append("};\n\n[maxvertexcount(").Append(maxVertices.ToString(CultureInfo.InvariantCulture))
            .Append(")]\nvoid main(triangle SPIRV_Cross_Input stage_input[3]");
        foreach (Element element in elements)
            if (element.IsPrimitiveId) text.Append(", uint primitive_id : ").Append(element.SourceSemantic);
        return new(text.Append(")\n{\n    return;\n}\n").ToString(), elements);
    }

    public static IReadOnlyList<Element>? ReadVertexOutputs(ReadOnlySpan<byte> binary)
        => TryGetChunks(binary, out _, out var output, out var program) && Read(program, 0) == VertexShaderModel50
            && program.Length % 4 == 0 && Read(program, 4) == program.Length / 4
            && TryReadSignature(output, isInput: false, out var elements) ? elements : null;

    private static bool TryGetChunks(ReadOnlySpan<byte> binary, out ReadOnlySpan<byte> input, out ReadOnlySpan<byte> output,
        out ReadOnlySpan<byte> program)
    {
        const int tableStart = 32;
        input = output = program = default;
        if (binary.Length < tableStart || !binary[..4].SequenceEqual("DXBC"u8)
            || Read(binary, 20) != 1 || Read(binary, 24) != binary.Length)
            return false;
        uint count = Read(binary, 28);
        if (count is < 3 or > 256 || tableStart + (count * 4) > binary.Length) return false;
        int end = tableStart + ((int)count * 4);
        for (int i = 0; i < count; i++)
        {
            uint offset = Read(binary, tableStart + (i * 4));
            if (offset < end || offset % 4 != 0 || offset > binary.Length - 8) return false;
            uint length = Read(binary, (int)offset + 4);
            int start = (int)offset + 8;
            if (length > binary.Length - start) return false;
            end = start + (int)length;
            ReadOnlySpan<byte> chunk = binary[start..end];
            ReadOnlySpan<byte> tag = binary.Slice((int)offset, 4);
            if (tag.SequenceEqual("ISGN"u8))
            {
                if (!input.IsEmpty || chunk.Length < 8) return false;
                input = chunk;
            }
            else if (tag.SequenceEqual("OSG5"u8) || tag.SequenceEqual("OSGN"u8))
            {
                if (!output.IsEmpty || chunk.Length < 8 || tag.SequenceEqual("OSG5"u8) && chunk.Length != 8) return false;
                output = chunk;
            }
            else if (tag.SequenceEqual("SHEX"u8) || tag.SequenceEqual("SHDR"u8))
            {
                if (!program.IsEmpty || chunk.Length < 8) return false;
                program = chunk;
            }
            else if (tag.SequenceEqual("ISG1"u8) || tag.SequenceEqual("OSG1"u8) || tag.SequenceEqual("DXIL"u8))
                return false;
        }
        return !input.IsEmpty && !output.IsEmpty && !program.IsEmpty;
    }

    private static bool TryReadSignature(ReadOnlySpan<byte> signature, bool isInput, out List<Element> elements)
    {
        const int headerSize = 8, elementSize = 24;
        elements = new();
        uint count = Read(signature, 0);
        if (count > 33 || Read(signature, 4) != headerSize || headerSize + (count * elementSize) > signature.Length)
            return false;
        int namesStart = headerSize + ((int)count * elementSize);
        var semantics = new HashSet<(string, uint)>();
        for (int i = 0; i < count; i++)
        {
            int record = headerSize + (i * elementSize);
            uint nameOffset = Read(signature, record);
            uint index = Read(signature, record + 4);
            uint system = Read(signature, record + 8);
            uint componentType = Read(signature, record + 12);
            uint register = Read(signature, record + 16);
            uint flags = Read(signature, record + 20);
            uint mask = flags & 15;
            uint expectedFlags = isInput ? mask : mask | ((15 ^ mask) << 8);
            if (nameOffset < namesStart || nameOffset >= signature.Length || mask is not (1 or 3 or 7 or 15)
                || flags != expectedFlags)
                return false;
            ReadOnlySpan<byte> name = signature[(int)nameOffset..];
            int terminator = name.IndexOf((byte)0);
            if (terminator <= 0) return false;
            string semantic = Encoding.ASCII.GetString(name[..terminator]);
            if (!semantics.Add((semantic.ToUpperInvariant(), index))) return false;
            if (system == PrimitiveId)
            {
                if (!isInput || i != count - 1 || index != 0 || componentType != UInt32 || register != uint.MaxValue || mask != 1
                    || !semantic.Equals("SV_PrimitiveID", StringComparison.OrdinalIgnoreCase))
                    return false;
            }
            else if (register != i || register >= 32 || (system switch
            {
                Position => index != 0 || componentType != Float32 || mask != 15
                    || !semantic.Equals("SV_Position", StringComparison.OrdinalIgnoreCase),
                0 => !Regex.IsMatch(semantic, @"\A[A-Za-z_](?:[A-Za-z0-9_]*[A-Za-z_])?\z")
                    || semantic.StartsWith("SV_", StringComparison.OrdinalIgnoreCase),
                _ => true,
            }))
                return false;
            string? scalar = componentType switch { UInt32 => "uint", SInt32 => "int", Float32 => "float", _ => null };
            if (scalar is null) return false;
            elements.Add(new(semantic, index, system, register, mask,
                scalar + (mask switch { 3 => "2", 7 => "3", 15 => "4", _ => "" })));
        }
        return true;
    }

    private static bool TryReadReturnOnly(ReadOnlySpan<byte> program, List<Element> inputs, out uint maxVertices)
    {
        maxVertices = 0;
        if (program.Length % 4 != 0 || Read(program, 0) != GeometryShaderModel50 || Read(program, 4) != program.Length / 4)
            return false;
        program = program[8..];
        if (!Take(ref program, Token(DeclareGlobalFlags, 1, AllowRefactoring))) return false;
        foreach (Element input in inputs)
        {
            uint operand = VertexInputOperand | (input.Mask << 4);
            bool matched = input.SystemValue switch
            {
                PrimitiveId => Take(ref program, Token(DeclareInput, 2), PrimitiveIdOperand),
                Position => Take(ref program, Token(DeclareInputSystemValue, 5), operand, TriangleVertices, input.Register, Position),
                _ => Take(ref program, Token(DeclareInput, 4), operand, TriangleVertices, input.Register),
            };
            if (!matched) return false;
        }
        if (!Take(ref program, Token(DeclareInputPrimitive, 1, TrianglePrimitive << 11))
            || !Take(ref program, Token(DeclareMaxOutputVertices, 2)) || program.Length != 8)
            return false;
        maxVertices = Read(program, 0);
        return maxVertices is > 0 and <= 1024 && Read(program, 4) == Token(Ret, 1);
    }

    private static bool Take(ref ReadOnlySpan<byte> data, params ReadOnlySpan<uint> words)
    {
        if (data.Length < words.Length * 4) return false;
        for (int i = 0; i < words.Length; i++)
            if (Read(data, i * 4) != words[i]) return false;
        data = data[(words.Length * 4)..];
        return true;
    }

    private static uint Read(ReadOnlySpan<byte> data, int offset) => BinaryPrimitives.ReadUInt32LittleEndian(data[offset..]);
    private static uint Token(uint opcode, uint length, uint controls = 0) => opcode | (length << 24) | controls;
}
