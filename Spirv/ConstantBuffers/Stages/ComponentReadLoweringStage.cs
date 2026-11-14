using Ruri.ShaderTools.Spirv;

namespace Ruri.ShaderTools.Spirv.ConstantBuffers.Stages;

/// <summary>
/// Turns "load the whole register, then extract component N" into "address
/// member N directly".
///
/// For every tracked load whose pointer is a retargeted chain, each downstream
/// <c>OpCompositeExtract</c> names a specific component; combined with the
/// chain's register that is a precise byte offset, which usually maps to a named
/// member. The extract is then replaced by a fresh access-chain + load pair that
/// reads the member directly, and — crucially — the new load INHERITS the
/// extract's result id, so every existing consumer keeps working untouched.
///
/// The bitcast hop (<c>Load v4float → Bitcast v4uint → CompositeExtract uint</c>)
/// is followed transparently; it is how bool members stored as uint are read.
/// The now-dead bitcast is left for the prune stage.
///
/// Loads with no component readers at all are simply re-typed in place, when
/// their chain translated to a vector-shaped member. That keeps the load valid
/// against the new pointer type without needing a downstream extract to hang the
/// rewrite on.
/// </summary>
internal static class ComponentReadLoweringStage
{
    public static void Run(StructuringContext context)
    {
        if (context.RetargetedChains.Count == 0)
        {
            return;
        }

        var trackedLoads = new Dictionary<uint, TrackedLoad>();

        // Bitcast result id → (its source load, the bitcast itself). SSA order
        // guarantees a bitcast follows its source load in module order, so this
        // single forward pass is sound.
        var bitcastSources = new Dictionary<uint, (TrackedLoad Load, SpirvInstruction Bitcast)>();

        var processedBitcasts = new Dictionary<uint, SpirvInstruction>();
        context.TrackedLoads = trackedLoads;
        context.ProcessedBitcasts = processedBitcasts;

        List<SpirvInstruction> instructions = context.Module.Instructions;
        var scalarDefinitions = Ruri.ShaderTools.Spirv.Analysis.ResultIdTable.Build(context.Module);

        for (int index = 0; index < instructions.Count; index++)
        {
            SpirvInstruction instruction = instructions[index];

            if (instruction.OpCode == SpvOpCode.OpLoad && instruction.WordCount >= 4
                && context.RetargetedChains.TryGetValue(instruction[3], out RetargetedChain? chain))
            {
                trackedLoads[instruction[2]] = new TrackedLoad
                {
                    Instruction = instruction,
                    ResultId = instruction[2],
                    OriginalResultTypeId = instruction[1],
                    HasComponentReaders = false,
                    AccessChain = chain,
                };
                continue;
            }

            if (instruction.OpCode == SpvOpCode.OpBitcast && instruction.WordCount >= 4
                && trackedLoads.TryGetValue(instruction[3], out TrackedLoad? bitcastSource))
            {
                bitcastSources[instruction[2]] = (bitcastSource, instruction);
                continue;
            }

            if (instruction.OpCode != SpvOpCode.OpCompositeExtract || instruction.WordCount < 5)
            {
                continue;
            }

            uint compositeId = instruction[3];
            TrackedLoad? load;

            if (trackedLoads.TryGetValue(compositeId, out TrackedLoad? direct))
            {
                load = direct;
            }
            else if (bitcastSources.TryGetValue(compositeId, out (TrackedLoad Load, SpirvInstruction Bitcast) viaBitcast))
            {
                load = viaBitcast.Load;
                processedBitcasts[compositeId] = viaBitcast.Bitcast;
            }
            else
            {
                continue;
            }

            load.HasComponentReaders = true;

            FlatAccessPath componentPath = load.AccessChain.OriginalAccessPath.With(instruction.Words[4..]);
            AccessTranslation? translation = AccessTranslator.Translate(load.AccessChain.Plan.Layout, componentPath, context.Constants);

            if (translation is null || !context.UniformPointerTypes.TryGetValue(translation.MemberTypeId, out uint pointerTypeId))
            {
                throw new InvalidOperationException("Admitted constant-buffer component no longer translates.");
            }

            // Capture before the extract is retired — the new load adopts its id.
            uint extractResultId = instruction[2];
            uint pointerResultId = context.Module.AllocateId();
            ushort chainOpCode = load.AccessChain.InstructionOpCode;

            index += DerivedIndexEmitter.Emit(context, instructions, index, translation);
            instructions.Insert(index, BuildChain(context.Module, chainOpCode, pointerTypeId, pointerResultId,
                load.AccessChain.BaseVariableId, translation.Indices));

            var scalarReads = LegacyDx11ConstantBufferSafety.ScalarRead(context.Module, scalarDefinitions,
                translation.MemberTypeId, instruction[1], extractResultId, pointerResultId);
            instructions.InsertRange(index + 1, scalarReads);
            // Skip the chain, typed load and optional bitcast, then retire the original extract.
            index += 1 + scalarReads.Count;
            instruction.MakeNop();
        }

        LegacyDx11ConstantBufferSafety.RestoreWholeScalarReads(context);
    }

    private static SpirvInstruction BuildChain(
        SpirvModule module, ushort opCode, uint pointerTypeId, uint resultId, uint baseVariableId, List<uint> indices)
    {
        int wordCount = 4 + indices.Count;
        Span<uint> words = wordCount <= 64 ? stackalloc uint[wordCount] : new uint[wordCount];

        words[0] = SpvOpCode.MakeInstructionWord(opCode, (ushort)wordCount);
        words[1] = pointerTypeId;
        words[2] = resultId;
        words[3] = baseVariableId;
        for (int i = 0; i < indices.Count; i++)
        {
            words[4 + i] = indices[i];
        }

        return module.CreateInstruction(opCode, words);
    }
}
