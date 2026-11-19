using System.Runtime.InteropServices;
using Ruri.ShaderTools.Spirv;
using Ruri.ShaderTools.Spirv.SymbolInjection;

namespace Ruri.ShaderTools.Pipeline.Naming;

// Unity 2022.3 HLSLSupport.cginc aliases these semantic tokens with macros.
// An OpName is an identifier, NOT its BuiltIn/Location semantic. Change only
// output identifiers before SPIRV-Cross emits both their declarations and uses.
internal static class LegacyDx11UnityIdentifierNaming
{
    private static readonly HashSet<string> Reserved = new(StringComparer.Ordinal)
    {
        "SV_Target", "SV_Target0", "SV_Target1", "SV_Target2", "SV_Target3",
        "SV_Target4", "SV_Target5", "SV_Target6", "SV_Target7", "SV_Depth",
        "SV_POSITION", "COLOR", "COLOR0", "COLOR1", "COLOR2", "COLOR3",
        "COLOR4", "COLOR5", "COLOR6", "COLOR7",
    };

    public static byte[] Apply(byte[] bytes)
    {
        // Parse otherwise accepts a malformed trailing instruction. Fail closed.
        var words = MemoryMarshal.Cast<byte, uint>(bytes.AsSpan());
        if (words.Length < SpirvModule.HeaderWordCount || bytes.Length % sizeof(uint) != 0)
            throw new ArgumentException("Unity identifier naming: malformed SPIR-V length.");
        var reader = new SpirvModule.InstructionReader(words);
        if (!reader.ReadToEnd())
            throw new ArgumentException("Unity identifier naming: malformed SPIR-V instruction.");
        SpirvModule module = SpirvModule.Parse(bytes);
        var outputs = module.Instructions.Where(i => i.OpCode == 59 && i.WordCount >= 4 && i[3] == 3)
            .Select(i => i[2]).ToHashSet();
        var allNames = new HashSet<string>(StringComparer.Ordinal);
        var names = new Dictionary<uint, List<string>>();
        foreach (SpirvInstruction i in module.Instructions)
        {
            if (i.OpCode is not (5 or 6)) continue;
            int slot = i.OpCode == 5 ? 2 : 3;
            if (i.WordCount <= slot) throw new ArgumentException("Unity identifier naming: malformed debug name.");
            string name = SpirvLiteral.ReadString(i.Words, slot);
            allNames.Add(name);
            if (i.OpCode != 5) continue;
            if (!names.TryGetValue(i[1], out var list)) names.Add(i[1], list = new());
            list.Add(name);
        }
        var patches = new List<NamePatch>();
        foreach (uint id in outputs.Order())
        {
            if (!names.TryGetValue(id, out var values) || !values.Any(Reserved.Contains)) continue;
            if (values.Count != 1)
                throw new InvalidOperationException("Unity identifier naming: ambiguous output name.");
            string replacement = "vrcr_io_" + id + "_" + values[0];
            if (!allNames.Add(replacement))
                throw new InvalidOperationException("Unity identifier naming: generated name collision.");
            patches.Add(new NamePatch(id, replacement));
        }
        // No semantic/decorations, uniforms, inputs, algorithm or resource binding edits.
        return DebugNameInjector.Inject(bytes, patches, Array.Empty<MemberNamePatch>());
    }
}
