using Ruri.ShaderTools.Pipeline.Backend;
using Ruri.ShaderTools.Spirv;
using Ruri.ShaderTools.Spirv.SymbolInjection;

namespace Ruri.ShaderTools.Pipeline.Naming;

internal static class HlslStructTypeNaming
{
    public static byte[] Disambiguate(byte[] spirv, IReadOnlyList<FlattenedBlock> blocks)
    {
        if (blocks.Count == 0) return spirv;

        var restored = blocks.SelectMany(block => block.AuthoredMembers.Select(member => (block.StructTypeId, member))).ToHashSet();
        var occupied = new HashSet<string>(StringComparer.Ordinal);
        var globals = new HashSet<string>(StringComparer.Ordinal);
        var structs = new Dictionary<uint, SpirvInstruction>();
        var arrays = new Dictionary<uint, uint>();
        var bufferTypes = new HashSet<uint>();
        var names = new Dictionary<uint, string>();

        foreach (SpirvInstruction instruction in SpirvModule.Parse(spirv).Instructions)
        {
            switch (instruction.OpCode)
            {
                case SpvOpCode.OpTypeStruct:
                    structs.Add(instruction[1], instruction);
                    break;
                case SpvOpCode.OpTypeArray or SpvOpCode.OpTypeRuntimeArray:
                    arrays.Add(instruction[1], instruction[2]);
                    break;
                case SpvOpCode.OpDecorate when instruction[2] is Decoration.Block or Decoration.BufferBlock:
                    bufferTypes.Add(instruction[1]);
                    break;
                case SpvOpCode.OpName:
                    string name = HlslIdentifier.Sanitize(SpirvLiteral.ReadString(instruction.Words, 2));
                    if (name.Length == 0) break;
                    occupied.Add(name);
                    names[instruction[1]] = name;
                    break;
                case SpvOpCode.OpMemberName:
                    string member = HlslIdentifier.Sanitize(SpirvLiteral.ReadString(instruction.Words, 3));
                    occupied.Add(member);
                    if (restored.Contains((instruction[1], instruction[2]))) globals.Add(member);
                    break;
            }
        }

        // SPIRV-Cross can keep the type name because it prefixes cbuffer members.
        // Restoring their external names later must not shadow the emitted type.
        var patches = new List<NamePatch>();
        foreach ((uint id, string name) in names)
        {
            if (structs.ContainsKey(id) && !bufferTypes.Contains(id) && globals.Contains(name))
                patches.Add(new NamePatch(id, ReserveAlias("type_" + name, id, occupied)));
        }

        // The backend uses the block type name for `cbuffer`. Structured buffer
        // members can depend on that declaration retaining its recovered binding name.
        foreach (var group in blocks.GroupBy(block => block.StructTypeId))
        {
            FlattenedBlock block = group.First();
            if (group.Skip(1).Any() || !bufferTypes.Contains(block.StructTypeId)
                || !HasStructMember(structs[block.StructTypeId], structs, arrays)
                || !names.TryGetValue(block.VariableId, out string? name) || globals.Contains(name)
                || names.Any(pair => pair.Key != block.VariableId && pair.Value == name)) continue;

            patches.Add(new NamePatch(block.VariableId, ReserveAlias("buffer_" + name, block.VariableId, occupied)));
            patches.Add(new NamePatch(block.StructTypeId, name));
        }
        return DebugNameInjector.Inject(spirv, patches, Array.Empty<MemberNamePatch>());
    }

    private static bool HasStructMember(SpirvInstruction block, Dictionary<uint, SpirvInstruction> structs, Dictionary<uint, uint> arrays)
    {
        foreach (uint type in block.Words[2..])
        {
            uint member = type;
            while (arrays.TryGetValue(member, out uint element)) member = element;
            if (structs.ContainsKey(member)) return true;
        }
        return false;
    }

    private static string ReserveAlias(string stem, uint id, HashSet<string> occupied)
    {
        string alias = stem;
        for (uint suffix = id; !occupied.Add(alias); suffix++) alias = $"{stem}_{suffix}";
        return alias;
    }
}
