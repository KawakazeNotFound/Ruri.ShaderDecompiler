using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Linq;

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
    public static byte[] Apply(byte[] bytes)
    {
        if (bytes.Length < 20 || bytes.Length % 4 != 0 || Read(bytes, 0) != 0x07230203)
            throw new ArgumentException("Malformed SPIR-V header or byte length.");
        var instructions = new List<(int Start, int Words, ushort Op)>();
        for (int offset = 20; offset < bytes.Length;)
        {
            uint header = Read(bytes, offset);
            int words = (int)(header >> 16);
            if (words < 1 || words > (bytes.Length - offset) / 4)
                throw new ArgumentException("Malformed SPIR-V instruction length.");
            // Production enables checked arithmetic. Bound the opcode before
            // narrowing; the packed header also contains the instruction length.
            instructions.Add((offset, words, (ushort)(header & 0xffffu)));
            offset += words * 4;
        }
        var entries = instructions.Where(i => i.Op == 15).ToList();
        if (!entries.Any(i => i.Words >= 4 && Read(bytes, i.Start + 4) == 3))
            return bytes;
        if (entries.Count != 1 || entries[0].Words < 4)
            throw new ArgumentException("Single-entry geometry module required for stream lowering.");
        var integerTypes = new HashSet<uint>();
        var constants = new Dictionary<uint, (uint Type, uint Value)>();
        foreach (var i in instructions)
        {
            if (i.Op == 21 && i.Words == 4 && Read(bytes, i.Start + 8) == 32 && Read(bytes, i.Start + 12) <= 1)
                if (!integerTypes.Add(Read(bytes, i.Start + 4)))
                    throw new ArgumentException("Duplicate SPIR-V integer type identity.");
            if (i.Op == 43 && i.Words == 4)
                if (!constants.TryAdd(Read(bytes, i.Start + 8), (Read(bytes, i.Start + 4), Read(bytes, i.Start + 12))))
                    throw new ArgumentException("Duplicate SPIR-V constant identity.");
            // Explicit output Stream decorations, including member decorations,
            // must agree with the stream-zero-only contract.
            if (i.Op == 71 && i.Words >= 3 && Read(bytes, i.Start + 8) == 29
                && (i.Words != 4 || Read(bytes, i.Start + 12) != 0))
                throw new ArgumentException("Nonzero or malformed geometry Stream decoration.");
            if (i.Op == 72 && i.Words >= 4 && Read(bytes, i.Start + 12) == 29
                && (i.Words != 5 || Read(bytes, i.Start + 16) != 0))
                throw new ArgumentException("Nonzero or malformed geometry member Stream decoration.");
        }
        var changes = instructions.Where(i => i.Op is 220 or 221).ToList();
        foreach (var i in changes)
        {
            if (i.Words != 2 || !constants.TryGetValue(Read(bytes, i.Start + 4), out var value)
                || !integerTypes.Contains(value.Type) || value.Value != 0)
                throw new ArgumentException("Geometry stream must be a proven int32/uint32 literal zero.");
        }
        if (changes.Count == 0) return bytes;
        // Validate every site before serializing anything. Keep instruction order,
        // constants, all output stores and all other instructions byte-for-byte.
        var result = new byte[bytes.Length - changes.Count * 4];
        bytes.AsSpan(0, 20).CopyTo(result);
        int cursor = 20;
        foreach (var i in instructions)
        {
            if (i.Op is 220 or 221)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(cursor, 4),
                    (1u << 16) | (uint)(i.Op == 220 ? 218 : 219));
                cursor += 4;
            }
            else
            {
                bytes.AsSpan(i.Start, i.Words * 4).CopyTo(result.AsSpan(cursor));
                cursor += i.Words * 4;
            }
        }
        return result;
    }

    private static uint Read(byte[] bytes, int offset)
        => BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset, 4));
}
