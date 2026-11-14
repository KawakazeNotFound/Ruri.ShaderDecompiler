using Ruri.ShaderTools.Spirv;
using Ruri.ShaderTools.Spirv.Analysis;

namespace Ruri.ShaderTools.Spirv.ConstantBuffers;

// Admission proof for a typed member replacing a raw register load. This is
// deliberately limited to exact loads or complete scalar-extract use graphs.
internal static class LegacyDx11ConstantBufferSafety
{
    public static bool PointerElementMatches(ResultIdTable definitions, SpirvInstruction chain, uint targetType)
        => definitions.TryGetDefinition(chain[1], out var pointer)
           && pointer.OpCode == SpvOpCode.OpTypePointer && pointer.WordCount == 4
           && pointer[3] == targetType;

    public static bool CanPreserveScalarType(ResultIdTable definitions, uint actual, uint expected)
        => actual == expected || IsScalar32(definitions, actual) && IsScalar32(definitions, expected);

    private static bool IsScalar32(ResultIdTable definitions, uint id)
        => definitions.TryGetDefinition(id, out var type)
           && (type.OpCode == SpvOpCode.OpTypeFloat && type.WordCount == 3 && type[2] == 32
               || type.OpCode == SpvOpCode.OpTypeInt && type.WordCount == 4 && type[2] == 32 && type[3] <= 1);

    public static bool CanReadScalarDirect(OperandUseIndex uses, LiveUseCounter live,
        ResultIdTable definitions, SpirvInstruction chain, uint targetType)
    {
        if (!IsScalar32(definitions, targetType)
            || !definitions.TryGetDefinition(chain[1], out var pointer)
            || pointer.OpCode != SpvOpCode.OpTypePointer || pointer.WordCount != 4
            || !IsScalar32(definitions, pointer[3])) return false;
        int loads = 0;
        var visited = new HashSet<SpirvInstruction>();
        foreach (var load in uses.UsersOf(chain[2]))
            if (visited.Add(load) && load.OpCode == SpvOpCode.OpLoad && load.WordCount == 4
                && load[3] == chain[2] && load[1] == pointer[3]) loads++;
        return loads > 0 && loads == live.ConsumerCount(chain[2]);
    }

    public static void RestoreWholeScalarReads(StructuringContext context)
    {
        var definitions = ResultIdTable.Build(context.Module);
        var tracked = context.TrackedLoads.Values.ToDictionary(load => load.Instruction);
        // Walk backwards so each original instruction is visited once despite
        // inserting its restore-bitcast immediately afterwards.
        for (int index = context.Module.Instructions.Count - 1; index >= 0; index--)
        {
            var instruction = context.Module.Instructions[index];
            if (!tracked.TryGetValue(instruction, out var load) || load.HasComponentReaders
                || load.AccessChain.Translation is not { } translation
                || instruction.OpCode != SpvOpCode.OpLoad) continue;
            uint originalType = load.OriginalResultTypeId, targetType = translation.MemberTypeId;
            if (originalType == targetType) continue;
            if (!IsScalar32(definitions, originalType) || !IsScalar32(definitions, targetType))
                throw new InvalidOperationException("Admitted whole constant-buffer read changed unsupported shape.");
            uint originalId = load.ResultId;
            var reads = ScalarRead(context.Module, definitions, targetType, originalType, originalId, instruction[3]);
            instruction.SetWords(reads[0].Words);
            context.Module.Instructions.Insert(index + 1, reads[1]);
            load.ResultId = instruction[2];
            // Dead-code pruning must retire the restore before its typed load.
            context.ProcessedBitcasts[originalId] = reads[1];
        }
    }

    private static bool Vector32(ResultIdTable definitions, uint id, out uint scalar, out uint count)
    {
        scalar = count = 0;
        if (!definitions.TryGetDefinition(id, out var type) || type.OpCode != SpvOpCode.OpTypeVector
            || type.WordCount != 4 || type[3] < 2 || type[3] > 4 || !IsScalar32(definitions, type[2])) return false;
        scalar = type[2]; count = type[3]; return true;
    }

    public static bool CanLowerAllReads(StructuringContext context, OperandUseIndex uses,
        LiveUseCounter live, ResultIdTable definitions, uint chainId, BlockLayout layout, FlatAccessPath path)
    {
        int loads = 0;
        var visited = new HashSet<SpirvInstruction>();
        foreach (var load in uses.UsersOf(chainId))
        {
            if (!visited.Add(load) || load.OpCode != SpvOpCode.OpLoad || load.WordCount != 4 || load[3] != chainId) continue;
            loads++;
            if (!Vector32(definitions, load[1], out uint scalar, out uint count)) return false;
            bool foundExtract = false;
            int supported = 0;
            var readers = new HashSet<SpirvInstruction>();
            foreach (var reader in uses.UsersOf(load[2]))
            {
                if (!readers.Add(reader)) continue;
                if (reader.OpCode == SpvOpCode.OpCompositeExtract && reader.WordCount == 5 && reader[3] == load[2])
                {
                    if (!Extract(context, definitions, layout, path, reader, scalar, count)) return false;
                    foundExtract = true; supported++;
                }
                else if (reader.OpCode == SpvOpCode.OpBitcast && reader.WordCount == 4 && reader[3] == load[2])
                {
                    if (!Vector32(definitions, reader[1], out uint bitScalar, out uint bitCount) || bitCount != count) return false;
                    int bitReaders = 0;
                    var bitVisited = new HashSet<SpirvInstruction>();
                    foreach (var extract in uses.UsersOf(reader[2]))
                    {
                        if (!bitVisited.Add(extract) || extract.OpCode != SpvOpCode.OpCompositeExtract
                            || extract.WordCount != 5 || extract[3] != reader[2]) continue;
                        if (!Extract(context, definitions, layout, path, extract, bitScalar, bitCount)) return false;
                        foundExtract = true; bitReaders++;
                    }
                    if (bitReaders == 0 || bitReaders != live.ConsumerCount(reader[2])) return false;
                    supported++;
                }
            }
            // Count only genuine operand edges. OperandUseIndex intentionally
            // also indexes literals; they neither authorize nor veto this proof.
            if (!foundExtract || supported != live.ConsumerCount(load[2])) return false;
        }
        return loads > 0 && loads == live.ConsumerCount(chainId);
    }

    private static bool Extract(StructuringContext context, ResultIdTable definitions,
        BlockLayout layout, FlatAccessPath path, SpirvInstruction extract, uint sourceScalar, uint lanes)
    {
        if (extract[1] != sourceScalar || extract[4] >= lanes) return false;
        var translated = AccessTranslator.Translate(layout, path.With(extract.Words[4..]), context.Constants);
        return translated is not null && CanPreserveScalarType(definitions, translated.MemberTypeId, extract[1]);
    }

    public static List<SpirvInstruction> ScalarRead(SpirvModule module, ResultIdTable definitions,
        uint memberType, uint expectedType, uint resultId, uint pointerId)
    {
        if (!CanPreserveScalarType(definitions, memberType, expectedType))
            throw new InvalidOperationException("Unsupported constant-buffer scalar bitcast shape.");
        uint loadId = memberType == expectedType ? resultId : module.AllocateId();
        var result = new List<SpirvInstruction>
        {
            module.CreateInstruction(SpvOpCode.OpLoad, [SpvOpCode.MakeInstructionWord(SpvOpCode.OpLoad, 4), memberType, loadId, pointerId])
        };
        if (memberType != expectedType)
            result.Add(module.CreateInstruction(SpvOpCode.OpBitcast,
                [SpvOpCode.MakeInstructionWord(SpvOpCode.OpBitcast, 4), expectedType, resultId, loadId]));
        return result;
    }
}
