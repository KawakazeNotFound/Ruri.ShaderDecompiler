using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Linq;

namespace Ruri.ShaderTools.Pipeline;

/// <summary>
/// spirv-cross HLSL SM50 throws "Unsupported builtin in HLSL." when a geometry
/// shader declares an Input variable with BuiltIn Position (0). Lower that
/// single vertex-array BuiltIn Position input to a fresh unused Location
/// decoration in a temporary SPIR-V buffer for emission, then restore its
/// field semantic inside struct SPIRV_Cross_Input back to SV_Position.
/// </summary>
internal static class LegacyDx11GeometryPositionInput
{
    private const ushort OpEntryPoint = 15;
    private const ushort OpTypeFloat = 22;
    private const ushort OpTypeVector = 23;
    private const ushort OpTypeArray = 28;
    private const ushort OpTypeStruct = 30;
    private const ushort OpTypePointer = 32;
    private const ushort OpVariable = 59;
    private const ushort OpDecorate = 71;
    private const ushort OpMemberDecorate = 72;

    private const uint ExecutionModelGeometry = 3;
    private const uint StorageClassInput = 1;
    private const uint DecorationLocation = 30;
    private const uint DecorationBuiltIn = 11;
    private const uint BuiltInPosition = 0;

    public static (byte[] Spirv, uint? SentinelLocation) Lower(byte[] bytes)
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
            instructions.Add((offset, words, (ushort)(header & 0xffffu)));
            offset += words * 4;
        }

        var entries = instructions.Where(i => i.Op == OpEntryPoint).ToList();
        if (!entries.Any(i => i.Words >= 4 && Read(bytes, i.Start + 4) == ExecutionModelGeometry))
            return (bytes, null);
        if (entries.Count != 1 || entries[0].Words < 4)
            throw new InvalidOperationException("Single-entry geometry module required for BuiltIn Position input lowering.");

        var float32Types = new HashSet<uint>();
        var vec4Types = new HashSet<uint>();
        var arrayElementTypes = new Dictionary<uint, uint>();
        var structTypes = new HashSet<uint>();
        var inputPointerPointees = new Dictionary<uint, uint>();
        var inputVariablePointees = new Dictionary<uint, uint>();
        var builtInPositionMemberStructs = new HashSet<uint>();
        var builtInPositionDecorations = new List<(int Start, uint Target)>();
        var inputLocations = new HashSet<uint>();
        var allLocations = new Dictionary<uint, List<uint>>();
        uint maxLocation = 0;
        bool hasAnyLocation = false;

        foreach (var inst in instructions)
        {
            if (inst.Op == OpTypeFloat && inst.Words >= 3 && Read(bytes, inst.Start + 8) == 32)
            {
                float32Types.Add(Read(bytes, inst.Start + 4));
            }
            else if (inst.Op == OpTypeVector && inst.Words == 4
                && float32Types.Contains(Read(bytes, inst.Start + 8))
                && Read(bytes, inst.Start + 12) == 4)
            {
                vec4Types.Add(Read(bytes, inst.Start + 4));
            }
            else if (inst.Op == OpTypeArray && inst.Words == 4)
            {
                arrayElementTypes[Read(bytes, inst.Start + 4)] = Read(bytes, inst.Start + 8);
            }
            else if (inst.Op == OpTypeStruct && inst.Words >= 2)
            {
                structTypes.Add(Read(bytes, inst.Start + 4));
            }
            else if (inst.Op == OpTypePointer && inst.Words == 4 && Read(bytes, inst.Start + 8) == StorageClassInput)
            {
                inputPointerPointees[Read(bytes, inst.Start + 4)] = Read(bytes, inst.Start + 12);
            }
            else if (inst.Op == OpVariable && inst.Words >= 4 && Read(bytes, inst.Start + 12) == StorageClassInput)
            {
                uint pointerTypeId = Read(bytes, inst.Start + 4);
                uint varId = Read(bytes, inst.Start + 8);
                if (!inputPointerPointees.TryGetValue(pointerTypeId, out uint pointeeTypeId))
                    throw new InvalidOperationException("Geometry Input OpVariable does not reference an Input OpTypePointer.");
                inputVariablePointees[varId] = pointeeTypeId;
            }
            else if (inst.Op == OpDecorate && inst.Words == 4)
            {
                uint target = Read(bytes, inst.Start + 4);
                uint decoration = Read(bytes, inst.Start + 8);
                uint value = Read(bytes, inst.Start + 12);
                if (decoration == DecorationBuiltIn && value == BuiltInPosition)
                {
                    builtInPositionDecorations.Add((inst.Start, target));
                }
                else if (decoration == DecorationLocation)
                {
                    if (!allLocations.TryGetValue(target, out var locs))
                    {
                        locs = new List<uint>();
                        allLocations[target] = locs;
                    }
                    locs.Add(value);
                }
            }
            else if (inst.Op == OpMemberDecorate && inst.Words == 5
                && Read(bytes, inst.Start + 12) == DecorationBuiltIn
                && Read(bytes, inst.Start + 16) == BuiltInPosition)
            {
                builtInPositionMemberStructs.Add(Read(bytes, inst.Start + 4));
            }
        }

        foreach (var kv in inputVariablePointees)
        {
            uint pointee = kv.Value;
            uint candidateStruct = arrayElementTypes.TryGetValue(pointee, out uint elem) ? elem : pointee;
            if (structTypes.Contains(candidateStruct) && builtInPositionMemberStructs.Contains(candidateStruct))
                throw new InvalidOperationException("Unsupported geometry Input struct block carrying BuiltIn Position.");
            if (allLocations.TryGetValue(kv.Key, out var locs))
            {
                foreach (uint loc in locs)
                {
                    inputLocations.Add(loc);
                    if (!hasAnyLocation || loc > maxLocation)
                    {
                        maxLocation = loc;
                        hasAnyLocation = true;
                    }
                }
            }
        }

        var inputBuiltInPositions = builtInPositionDecorations
            .Where(d => inputVariablePointees.ContainsKey(d.Target))
            .ToList();

        if (inputBuiltInPositions.Count == 0)
            return (bytes, null);
        if (inputBuiltInPositions.Count != 1)
            throw new InvalidOperationException("Multiple geometry Input BuiltIn Position variables are not supported.");

        var match = inputBuiltInPositions[0];
        if (allLocations.ContainsKey(match.Target))
            throw new InvalidOperationException("Geometry Input BuiltIn Position variable already carries a Location decoration.");

        uint pointeeType = inputVariablePointees[match.Target];
        if (!arrayElementTypes.TryGetValue(pointeeType, out uint elemType) || !vec4Types.Contains(elemType))
            throw new InvalidOperationException("Geometry Input BuiltIn Position variable must have array-of-float4 pointee type.");

        uint sentinelLocation = hasAnyLocation ? checked(maxLocation + 1) : 0u;
        while (inputLocations.Contains(sentinelLocation))
            sentinelLocation = checked(sentinelLocation + 1);

        var rewritten = (byte[])bytes.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(rewritten.AsSpan(match.Start + 8, 4), DecorationLocation);
        BinaryPrimitives.WriteUInt32LittleEndian(rewritten.AsSpan(match.Start + 12, 4), sentinelLocation);
        return (rewritten, sentinelLocation);
    }

    public static string RestoreSemantic(string source, uint sentinelLocation)
    {
        int structIdx = FindSingleStruct(source, "struct SPIRV_Cross_Input");
        int openBrace = source.IndexOf('{', structIdx);
        if (openBrace < 0)
            throw new InvalidOperationException("Missing opening brace for struct SPIRV_Cross_Input.");
        int closeBrace = source.IndexOf("};", openBrace + 1, StringComparison.Ordinal);
        if (closeBrace < 0)
            throw new InvalidOperationException("Missing closing brace for struct SPIRV_Cross_Input.");

        string body = source.Substring(openBrace, closeBrace - openBrace);
        string sentinelSemantic = $": TEXCOORD{sentinelLocation};";
        int first = body.IndexOf(sentinelSemantic, StringComparison.Ordinal);
        if (first < 0)
            throw new InvalidOperationException($"Emitted struct SPIRV_Cross_Input missing expected {sentinelSemantic} semantic.");
        if (body.IndexOf(sentinelSemantic, first + sentinelSemantic.Length, StringComparison.Ordinal) >= 0)
            throw new InvalidOperationException($"Emitted struct SPIRV_Cross_Input has duplicate {sentinelSemantic} semantic.");

        string updatedBody = body.Substring(0, first) + ": SV_Position;" + body.Substring(first + sentinelSemantic.Length);
        return source.Substring(0, openBrace) + updatedBody + source.Substring(closeBrace);
    }

    private static int FindSingleStruct(string source, string marker)
    {
        int found = -1;
        int searchFrom = 0;
        while (true)
        {
            int idx = source.IndexOf(marker, searchFrom, StringComparison.Ordinal);
            if (idx < 0) break;
            int end = idx + marker.Length;
            bool startOk = idx == 0 || source[idx - 1] is '\n' or '\r';
            bool endOk = end < source.Length && source[end] is ' ' or '\t' or '\r' or '\n' or '{';
            if (startOk && endOk)
            {
                if (found >= 0)
                    throw new InvalidOperationException($"Emitted HLSL has multiple {marker} declarations.");
                found = idx;
            }
            searchFrom = end;
        }
        if (found < 0)
            throw new InvalidOperationException($"Emitted HLSL missing {marker}.");
        return found;
    }

    private static uint Read(byte[] bytes, int offset)
        => BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset, 4));
}
