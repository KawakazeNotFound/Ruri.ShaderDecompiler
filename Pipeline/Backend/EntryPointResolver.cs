using System.Runtime.InteropServices;
using Ruri.ShaderTools.Spirv;

namespace Ruri.ShaderTools.Pipeline.Backend;

/// <summary>Which entry point to emit, and what it is.</summary>
internal readonly record struct EntryPointSelection(PipelineStage Stage, string? Name, uint ExecutionModel);

/// <summary>
/// Picks the entry point to hand the source backend.
///
/// Prefers the name the engine symbols asked for; falls back to the module's
/// first entry point. The RAW execution model travels with the selection because
/// the backend needs it to disambiguate entry points — the friendly stage name is
/// not enough when a module carries several.
/// </summary>
internal static class EntryPointResolver
{
    public static EntryPointSelection Resolve(byte[] spirv, string? preferredName)
    {
        ReadOnlySpan<uint> words = MemoryMarshal.Cast<byte, uint>(spirv.AsSpan());
        if (words.Length < SpirvModule.HeaderWordCount || words[0] != SpirvModule.MagicNumber)
            throw new ArgumentException("Invalid SPIR-V binary.", nameof(spirv));
        EntryPointSelection? first = null;

        var reader = new SpirvModule.InstructionReader(words);
        while (reader.MoveNext())
        {
            var instruction = reader.Current;
            if (reader.OpCode != SpvOpCode.OpEntryPoint || instruction.Length < 3)
            {
                continue;
            }

            // OpEntryPoint: [header, execution-model, function-id, name…, interface-ids…]
            uint executionModel = instruction[1];
            string name = SpirvLiteral.ReadString(instruction, 3);
            var candidate = new EntryPointSelection(ShaderStageClassifier.FromExecutionModel(executionModel), name, executionModel);

            first ??= candidate;

            if (!string.IsNullOrWhiteSpace(preferredName) && string.Equals(name, preferredName, StringComparison.Ordinal))
            {
                return candidate;
            }
        }

        return first ?? new EntryPointSelection(PipelineStage.Unknown, preferredName, 0u);
    }
}
